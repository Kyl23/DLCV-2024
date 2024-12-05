import os
import torchvision
from torchvision import models
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torchvision import transforms as T
import numpy as np
import imageio
import os
import random

def OriginTrans(x):
    return x

def OnlyInputTrans(TRANSFORM):
    def _only_input_trans(x):
        if x.shape == (3, 512, 512):
            x = TRANSFORM(x)

        return x

    return _only_input_trans

TRANSFORM_IMG = [
                    T.RandomHorizontalFlip(p=1), 
                    T.RandomVerticalFlip(p=1),
                    T.Compose([T.RandomHorizontalFlip(p=1), T.RandomVerticalFlip(p=1)]),
                    OnlyInputTrans(T.GaussianBlur(kernel_size=3)),
                    # OnlyInputTrans(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)),
                    T.Compose([T.RandomCrop((450, 450)), T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)]),
                    OriginTrans
                ]

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou


def cal_score(pred_path, ground_path):
    pred = read_masks(pred_path)
    labels = read_masks(ground_path)

    return mean_iou_score(pred, labels)

num_classes = 7
batch_size= 64

color_map = {
            0:  [0, 255, 255],
            1:  [255, 255, 0],
            2:  [255, 0, 255],
            3:  [0, 255, 0],
            4:  [0, 0, 255],
            5:  [255, 255, 255],
            6: [0, 0, 0],
        }

def rgb2num(rgb):
    return rgb[0] * 1000000 + rgb[1] * 1000 + rgb[2]

reversed_color_map = {}
for key, value in color_map.items():
    reversed_color_map[rgb2num(value)] = key
    
class ImgDataset(Dataset):
    # data loading
    def __init__(self, root_dir, data_type, from_cache=None):
        assert data_type in ("train", "valid")
        self.path = []
        self.data_type = data_type
        self.num_classes = num_classes
        self.cache = {}
        self.from_cache = from_cache
        
        for i in os.listdir(root_dir):
            if i.endswith("sat.jpg"):
                self.path.append(os.path.join(root_dir, i))
        
        if from_cache and os.path.exists(from_cache): # read from cache if cache existed
            import pickle
            with open(self.from_cache, 'rb') as file:
                self.cache = pickle.load(file)
                print(f"loaded cache from: {from_cache}")

    def process_label(self, mask):
        y = torch.zeros(mask.shape[1:])
        keys = mask.permute(1, 2, 0).reshape(-1, 3).numpy()
        keys = [rgb2num(k) for k in keys]
        
        y = torch.tensor([reversed_color_map[k] for k in keys]).reshape(mask.shape[1:])
        
        return y
    
    def read_image(self, path):
        x = torchvision.io.read_image(path)
        y = torchvision.io.read_image(path.replace("sat.jpg", "mask.png"))
        y = self.process_label(y)
        
        return x.float(), y

    # working for indexing
    def __getitem__(self, index):
        NORMALIZE = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        if self.data_type == "train":
            img_index = index // len(TRANSFORM_IMG)
            aug_type = index % len(TRANSFORM_IMG)
            path = self.path[img_index]
            
            if path not in self.cache:
                x, y = self.read_image(path)
                self.cache[path] = [
                    x,
                    y,
                    path.replace("sat.jpg", "mask.png")
                ]
            
            data = self.cache[path].copy()

            seed = random.randint(0,1000)
            torch.manual_seed(seed)
            data[0] = TRANSFORM_IMG[aug_type](data[0])
            torch.manual_seed(seed)
            data[1] = TRANSFORM_IMG[aug_type](data[1].unsqueeze(0)).squeeze(0)
        else:
            path = self.path[index]
            if path not in self.cache:
                x, y = self.read_image(path)
                self.cache[path] = [
                    x,
                    y,
                    path.replace("sat.jpg", "mask.png")
                ]
            
            data = self.cache[path].copy()
        
        data[0] = NORMALIZE(data[0])
        return data

    # return the length of our dataset
    def __len__(self):
        return len(self.path) * len(TRANSFORM_IMG) if self.data_type == "train" else len(self.path)

def revert_seg_img2img(seg_img):
    flat_seg = seg_img.view(-1)
    
    colors = torch.tensor([color_map[i] for i in range(len(color_map))], dtype=torch.uint8)
    img = colors[flat_seg].view(*seg_img.shape, 3).permute(0, 3, 1, 2)
    
    return img

def criterion(a, f):
    return nn.CrossEntropyLoss()(a, f)

def evaluate(model, valid_loader, data_type):
    os.system("rm p2_data/pred/*")
    model.eval()

    pbar = tqdm(valid_loader)
    pbar.set_description(f"Evaluating")

    for (x, _, paths) in pbar:
        x = x.cuda()
        
        with autocast('cuda'):
            with torch.no_grad():
                output = model(x)
            
        seg_img = torch.argmax(output, dim=1)
        imgs = revert_seg_img2img(seg_img.cpu().detach())
        
        for path, img in zip(paths, imgs):
            torchvision.io.write_png(img, path.replace(data_type, 'pred'))
    
    model.train()
    return cal_score("p2_data/pred/", f"p2_data/{data_type}")

train_loader = DataLoader(ImgDataset(os.path.join("p2_data", "train"),  "train", 'cache.pkl'), batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
valid_loader = DataLoader(ImgDataset(os.path.join("p2_data", "validation"), "valid", 'cache_test.pkl'), batch_size=batch_size, num_workers=8, shuffle=False)#, num_workers=1, persistent_workers=True)

def train(model, epochs=50, lr=0.01, model_prefix="", run_parallel=False):
    scaler = GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.8, patience=3, min_lr=1e-6, verbose=1)
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

        for i, (x, y, _) in enumerate(pbar):
            i += 1
            step += 1

            x = x.cuda()
            y = y.cuda()

            with autocast('cuda'):
                outputs = model(x)
                loss = criterion(outputs, y)

            if run_parallel:
                loss = loss.mean()

            # opt.zero_grad()
            # loss.backward()
            # opt.step()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss = f"{total_loss / i:.4f}")

            # if step % 2000 == 0:
            #   torch.save(resnet.state_dict(), f'./{model_prefix}_{epoch}_{i}_{step}.pt')
        if epoch % 4 == 0:
            accuracy = evaluate(model, train_loader, 'train')
        accuracy = evaluate(model, valid_loader, 'validation')
        # scheduler.step(accuracy)
        if accuracy >= history_accuracy:
            history_accuracy = accuracy
            print(f"new high accuracy: {history_accuracy}")
            torch.save(model, f'./{model_prefix}_best_model.pt')
          
    torch.save(model, f'./{model_prefix}.pt')

class VGG_FCN32(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.vgg_features = models.vgg16(pretrained=True).features
        self.fcn32_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fcn32_upsampling = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.vgg_features(x)
        x = self.fcn32_conv(x)
        x = self.fcn32_upsampling(x)
        
        return x
    
model = VGG_FCN32(num_classes).cuda()

# print(model)
train(model, 20, 5e-5, "segmentation_vgg16", True)
