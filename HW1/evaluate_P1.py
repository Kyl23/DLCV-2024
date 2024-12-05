import torch
import sys
import pandas as pd
import torchvision
from torchvision import transforms
import os
from train_utils import *
from torch.utils.data import Dataset, DataLoader

template_path, root_dir, output_path= sys.argv[1],sys.argv[2],sys.argv[3]

model = torch.load("P1_utils/setting_c_best_model.pt").cuda()
batch_size = 16
template = pd.read_csv(template_path)

out = [
    [],
    [],
    []
]

model.eval()

class ImgDataset(Dataset):
    # data loading
    def __init__(self, root_dir, files, id):
        self.x = files
        self.root_dir = root_dir
        self.ids = id
        

    def read_image(self, path):
        x = torchvision.io.read_image(os.path.join(self.root_dir, path))

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
        
        img_path = self.x[index]
        x = self.read_image(img_path)

        return self.ids[index], img_path, NORMALIZE(x)

    # return the length of our dataset
    def __len__(self):
        return len(self.x)

valid_loader = DataLoader(ImgDataset(root_dir, template['filename'], template['id']), batch_size=batch_size, shuffle=False, num_workers=2)

for i, f, x in valid_loader:
    NORMALIZE = transforms.Compose([
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    transforms.Resize([128,128])
                ])
    
    x = x.cuda()    
    with torch.no_grad():
        output = model(x)
    
    labels = torch.argmax(output, dim=1).cpu().tolist()
    
    i = i.tolist()
    for t in zip(i, f, labels):
        out[0].append(t[0])
        out[1].append(t[1])
        out[2].append(t[2])

df = pd.DataFrame({
    'id': out[0],
    'filename': out[1],
    'label': out[2]
})

df.to_csv(output_path, index=False)