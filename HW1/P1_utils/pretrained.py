import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

lr = 1e-5
batch_size = 64

class ImgDataset(Dataset):
    # data loading
    def __init__(self, root_dir):
        self.cache = {}
        self.x = []
        for i in os.listdir(root_dir):
          self.x.append(os.path.join(root_dir, i))
        # self.y = []
    def read_image(self, path):
      x = torchvision.io.read_image(path).float()
      x = TRANSFORM_IMG(x)

      return x

    # working for indexing
    def __getitem__(self, index):
        if self.x[index] not in self.cache:
            self.cache[self.x[index]] = self.read_image(self.x[index])

        return self.cache[self.x[index]]

    # return the length of our dataset
    def __len__(self):
        return len(self.x)

images_loader = DataLoader(ImgDataset(os.path.join("p1_data", "mini", "train")), batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)


def train(from_pretrained=None, start=0, epochs=100):
  resnet = torchvision.models.resnet50(pretrained=False)

  if from_pretrained:
    resnet.load_state_dict(torch.load(from_pretrained))
    print("loaded pretrained")

  from byol_pytorch import BYOL

  learner = BYOL(
      resnet,
      image_size = 128,
      hidden_layer = 'avgpool'
  )

  opt = torch.optim.Adam(learner.parameters(), lr=lr)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=3, min_lr=1e-6, verbose=1)
  scaler = GradScaler()

  from tqdm import tqdm

  learner = learner.to('cuda')

  for epoch in range(start, start + epochs):
    epoch += 1
    pbar = tqdm(images_loader)
    pbar.set_description(f"Training epoch [{epoch}/{start + epochs}]")
    total_loss = 0

    for i, images in enumerate(pbar):
      i += 1
      images = images.cuda()

      with autocast('cuda'):
          outputs_loss = learner(images)
      opt.zero_grad()
      loss = outputs_loss
      scaler.scale(loss).backward()
      scaler.step(opt)
      scaler.update()

      learner.update_moving_average() # update moving average of target encoder
      total_loss += loss.item()
      pbar.set_postfix(loss = f"{total_loss / i:.4f}")
  #  scheduler.step(total_loss / i)
    # if epoch % 10 == 0:
    #   torch.save(resnet.state_dict(), f'./pretrained_{epoch}.pt')

  torch.save(resnet.state_dict(), f'./pretrained_{epoch}.pt')

train(None, 0, 100)
train("pretrained_100.pt", 100, 100)
