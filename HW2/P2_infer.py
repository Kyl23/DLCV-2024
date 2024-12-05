# %%
from UNet import *
from utils import *

import torch
import sys

# %%
model = UNet()
model.load_state_dict(torch.load(sys.argv[3]))
model = model.cuda()

# %%
import os
from tqdm import tqdm
from torchvision.utils import save_image

noise_dir = sys.argv[1]
DDIM_STEP = 50
DDPM_STEP = 1000

STEP = -(DDPM_STEP // DDIM_STEP)


beta = beta_scheduler(1001)
alpha = 1 - beta
alpha_cum = torch.cumprod(alpha, dim=0)

if not os.path.exists(sys.argv[2]):
    os.mkdir(sys.argv[2])
    
def generate(eta=0.0):
    for path in os.listdir(noise_dir):
        path = os.path.join(noise_dir, path)

        img = torch.load(path)
        
        for t in tqdm(range(DDPM_STEP + STEP + 1, STEP + 1, STEP)):
            with torch.no_grad():
                eps = model(img.cuda(), torch.tensor(t).cuda())

            # alpha_t = torch.full((img.shape[0], 1, 1, 1), (1 - beta_scheduler(1001)[t])).cuda()
            # if t + STEP <= DDPM_STEP:
            #     alpha_t_minus_1 = torch.full((img.shape[0], 1, 1, 1), (1 - beta_scheduler(1001)[t + STEP])).cuda()
            # else:
            #     alpha_t_minus_1 = torch.full((img.shape[0], 1, 1, 1), 1.0).cuda()

            alpha_t = alpha_cum[t]
            alpha_t_minus_1 = alpha_cum[t + STEP if t > -STEP else 0] 
            # alpha_t_minus_1 = alpha_cum[t + STEP] if t > -STEP else torch.tensor(1)

            sigma_t = eta * torch.sqrt((1 - alpha_t_minus_1) / (1 - alpha_t) * (1 - alpha_t / alpha_t_minus_1))

            random_noise = sigma_t * torch.randn_like(img)
            to_xt = (torch.sqrt(1 - alpha_t_minus_1 - sigma_t ** 2) * eps)
            pred_x0 = ((img - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t))

            # pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            img = torch.sqrt(alpha_t_minus_1) * pred_x0 + to_xt + random_noise

            # img =  torch.sqrt(alpha_t_minus_1 / alpha_t) * img - torch.sqrt(alpha_t_minus_1 * ( 1 - alpha_t) / alpha_t) * eps + torch.sqrt(1 - alpha_t_minus_1 - sigma_t ** 2) * eps + sigma_t * torch.randn_like(img)
        # img = torch.clamp(img, -1.0, 1.0)
        min_value = img.min()
        img = img - min_value
        max_value = img.max()
        img /= max_value

        save_image(img, os.path.join(sys.argv[2], f"{eta}_"+ os.path.basename(path).replace('.pt', ".png")))

# %% [markdown]
# ## Generate eta 0

# %%
generate(0)
