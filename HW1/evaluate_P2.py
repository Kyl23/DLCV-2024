import os
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torchvision import transforms as T
import os
import sys
from torch import nn
from torchvision import models

batch_size = 8

color_map = {
            0:  [0, 255, 255],
            1:  [255, 255, 0],
            2:  [255, 0, 255],
            3:  [0, 255, 0],
            4:  [0, 0, 255],
            5:  [255, 255, 255],
            6: [0, 0, 0],
        }

class ImgDataset(Dataset):
    # data loading
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.path = os.listdir(root_dir)
    
    def read_image(self, path):
        x = torchvision.io.read_image(os.path.join(self.root_dir, path))
        
        return x.float()

    def __getitem__(self, index):
        NORMALIZE = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        return NORMALIZE(self.read_image(self.path[index])), (self.path[index][:-7] + "mask.png")

    def __len__(self):
        return len(self.path)

valid_loader = DataLoader(ImgDataset(sys.argv[1]), batch_size=batch_size, num_workers=2, shuffle=False)

def revert_seg_img2img(seg_img):
    flat_seg = seg_img.view(-1)
    
    colors = torch.tensor([color_map[i] for i in range(len(color_map))], dtype=torch.uint8)
    img = colors[flat_seg].view(*seg_img.shape, 3).permute(0, 3, 1, 2)
    
    return img

def evaluate(model, valid_loader):
    model.eval()

    for x, paths in valid_loader:
        x = x.cuda()
        
        with autocast('cuda'):
            with torch.no_grad():
                output = model(x)
            
        seg_img = torch.argmax(output, dim=1)
        imgs = revert_seg_img2img(seg_img.cpu().detach())
        
        for path, img in zip(paths, imgs):
            torchvision.io.write_png(img, os.path.join(sys.argv[2], path))

class DeepLab(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, n_classes, 1, 1, 0)
        self.model.aux_classifier[4] = nn.Conv2d(256, n_classes, 1, 1, 0)

    def forward(self, x):
        x = self.model(x)['out']
    
        return x

model = torch.load("P2_utils/deeplabv3_resnet101_20_best_model.pt").cuda()

evaluate(model, valid_loader)