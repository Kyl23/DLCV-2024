import warnings
warnings.filterwarnings("ignore")

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torchvision
from torchvision import transforms
from torch.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm
from torch import nn

epochs = 50
lr = 3e-4
batch_size=16

"""### P1 finetune resnet50 backbone + classifier"""

def OriginTrans(x):
    return x

TRANSFORM_IMG = [
                    transforms.RandomHorizontalFlip(p=1),  # 隨機水平翻轉
                    transforms.RandomVerticalFlip(p=1),
                    transforms.Compose([transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1)]),
                    transforms.RandomRotation(degrees=40),
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    OriginTrans
                ]

class ImgDataset(Dataset):
    # data loading
    def __init__(self, root_dir, data_type, num_classes, from_cache=None):
        assert data_type in ("train", "val")
        self.data_type = data_type
        self.num_classes = num_classes
        self.x = []
        self.y = []
        self.from_cache = from_cache
        self.cache = {}
        file_map_label = {}

        label_path = os.path.join(root_dir, f"{data_type}.csv")
        data = pd.read_csv(label_path)

        for x, y in zip(data['filename'].tolist(), data['label'].tolist()):
            file_map_label[x] = int(y)

        root_dir = os.path.join(root_dir, data_type)

        for i in os.listdir(root_dir):
            self.x.append(os.path.join(root_dir, i)) # save image path
            self.y.append(file_map_label[i]) # save image label

        if self.from_cache and os.path.exists(self.from_cache): # read from cache if cache existed
            import pickle
            with open(self.from_cache, 'rb') as file:
                self.cache = pickle.load(file)
                print(f"{self.data_type} loaded cache from: {self.from_cache}")

    def read_image(self, path):
        x = torchvision.io.read_image(path)

        # padding image to square
        if x.shape[1] != x.shape[2]:
            max_bound = max(x.shape[1:])

            new_x = torch.zeros((3, max_bound, max_bound))
            new_x[:, :x.shape[1], :x.shape[2]] = x
            x = new_x

        x = x.float()
        x = transforms.Resize((128,128))(x)

        return x

    # working for indexing
    def __getitem__(self, index):
        NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        if self.data_type == "train":
            img_index = index // len(TRANSFORM_IMG)
            aug_type = index % len(TRANSFORM_IMG)
            img_path = self.x[img_index]

            if img_path not in self.cache:
                self.cache[img_path] = self.read_image(img_path)

            return NORMALIZE(TRANSFORM_IMG[aug_type](self.cache[img_path])), self.y[img_index]
        else:
            img_path = self.x[index]
            if img_path not in self.cache:
                self.cache[img_path] = self.read_image(img_path)

            return NORMALIZE(self.cache[img_path]), self.y[index]


    # return the length of our dataset
    def __len__(self):
        return len(self.x) * len(TRANSFORM_IMG) if self.data_type == "train" else len(self.x)

num_classes = 65

train_loader = DataLoader(ImgDataset(os.path.join("p1_data", "office"), "train", num_classes, 'cache.pkl'), batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
valid_loader = DataLoader(ImgDataset(os.path.join("p1_data", "office"), "val", num_classes), batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

def criterion(a, f):
    return nn.CrossEntropyLoss()(a, f)

def evaluate(model, valid_loader):
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(valid_loader)
    pbar.set_description(f"Evaluating")

    for (x, y) in pbar:
        x = x.cuda()
        y = y.cuda()
        
        # with autocast('cuda'):
        with torch.no_grad():
            outputs = model(x)
            
        pred_label = torch.argmax(outputs, dim=1)
        label = y

        for (a, f) in zip(pred_label, label):
            if a == f:
                correct += 1
            total += 1

        accuracy = correct / total
        pbar.set_postfix(acc = f"{accuracy:.4f}")

    model.train()
    return accuracy

def train(model, epochs=50, lr=0.01, model_prefix="", run_parallel=False):
    scaler = GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.8, patience=3, min_lr=1e-6, verbose=1)
    history_accuracy = 0
    step = 0

    model = model.to('cuda')


    if run_parallel:
        model = nn.parallel.DataParallel(model)

    for epoch in range(epochs):
        epoch += 1

        pbar = tqdm(train_loader)
        pbar.set_description(f"Training epoch [{epoch}/{epochs}]")
        total_loss = 0

        for i, (x, y) in enumerate(pbar):
            i += 1
            step += 1

            x = x.cuda()
            y = y.cuda()

            # with autocast('cuda'):
            outputs = model(x)
            loss = criterion(outputs, y)

            if run_parallel:
                loss = loss.mean()

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss = f"{total_loss / i:.4f}")

        if model_prefix == "setting_c" and epoch == 1:
            torch.save(model, f'./{model_prefix}_{epoch}.pt')
        if epoch % 5 == 0:
            accuracy = evaluate(model, train_loader)
        accuracy = evaluate(model, valid_loader)
    #      scheduler.step(accuracy)
        if accuracy >= history_accuracy:
            history_accuracy = accuracy
            print(f"new high accuracy: {history_accuracy}")
            torch.save(model, f'./{model_prefix}_best_model.pt')
          
    torch.save(model, f'./{model_prefix}.pt')
#     return model

# TRANSFORM_IMG = None

class Model(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        num_features = backbone.fc.out_features

        self.fc = nn.Sequential(
#            nn.ReLU(),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        return x

