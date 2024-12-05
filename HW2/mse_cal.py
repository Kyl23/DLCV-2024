import torch
from torch import nn
import torchvision
import os

def load_images2tensor(root_dir):
    img_list = os.listdir(root_dir)
    img_list.sort()
    img = []
    for l in img_list:
        l = os.path.join(root_dir, l)
        i = torchvision.io.read_image(l)
        img.append(i.unsqueeze(0))

    return torch.cat(img)


import sys
GT = load_images2tensor(sys.argv[1])
Pd = load_images2tensor(sys.argv[2])

print(torch.mean(((Pd - GT) ** 2).float()), nn.MSELoss()(Pd.float(), GT.float()))