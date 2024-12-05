# %%
import os, sys
import torch
from omegaconf import OmegaConf
from tqdm import tqdm, trange
import time
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

# %%
import re

def split_text_by_token(text):
    pattern = r"(<new1>|<new2>)"
    split_text = re.split(pattern, text)
    result = [item for item in split_text if item]
    
    return result

def special_embedding(model, text):
    device = model.device
    embedding_layer = model.cond_stage_model.transformer.text_model.embeddings.token_embedding
    position_encoding_layer = model.cond_stage_model.transformer.text_model.embeddings.position_embedding
    tokenizer = model.cond_stage_model.tokenizer
    
    max_length = 77
    bos, eos = tokenizer.bos_token, tokenizer.eos_token
    bos = tokenizer(bos, return_tensors='pt', add_special_tokens=False).to(device)['input_ids']
    eos = tokenizer(eos, return_tensors='pt', add_special_tokens=False).to(device)['input_ids']
    
    embedded_text = [embedding_layer(bos)]
    
    text_list = split_text_by_token(text)
    new_tokens = ["<new1>", "<new2>"]
    for t in text_list:
        if t in new_tokens:
            embedded_text.append(string_to_param_dict[t].to(device).unsqueeze(0))
        else:
            tokenized_t = tokenizer(t, return_tensors='pt', add_special_tokens=False).to(device)['input_ids']
            embedded_text.append(embedding_layer(tokenized_t))
    # embedded_text.append(embedding_layer(eos))
    embedded_text += [embedding_layer(eos) for i in range(max_length)]
    
    embedded_text = torch.cat(embedded_text, dim=1)
    embedded_text = embedded_text.squeeze(0)

    embedded_text = embedded_text[:max_length]
    position_encoding = position_encoding_layer(torch.arange(0, max_length, 1).to(device))
    
    return (embedded_text + position_encoding).unsqueeze(0)

import json
def load_json(path):
    with open(path) as f:
        return json.load(f)      

import sys

seed_everything(42)
config = OmegaConf.load(f"v1-inference.yaml")
model = load_model_from_config(config, sys.argv[3])

string_to_param_dict = torch.load(f"models/P3.ckpt")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model = model.eval()

sampler = DPMSolverSampler(model)

# %%
template = load_json(sys.argv[1])
output_root_dir = sys.argv[2]

if not os.path.exists(output_root_dir):
    os.mkdir(output_root_dir)
C, H, f, W, ddim_steps, scale, ddim_eta = 4, 512, 8, 512, 50, 7.5, 0

n_samples = 1
# %%
import math
from torchvision.utils import save_image
from collections import defaultdict

total_gen = 25
noted = defaultdict(int)

for idx, value in reversed(template.items()):
    type_folder_path = os.path.join(output_root_dir, str(idx))
    if not os.path.exists(type_folder_path):
        os.mkdir(type_folder_path)

    # if idx == str(0):
    #     continue

    data = [[i] * n_samples for i in value['prompt']]

    start_code = None

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(math.ceil(25 / n_samples), desc="Sampling"):
                    for prompt_idx, prompts in enumerate(tqdm(data, desc="data")):
                        prompt_folder_path = os.path.join(type_folder_path, str(prompt_idx))
                        if not os.path.exists(prompt_folder_path):
                            os.mkdir(prompt_folder_path)
                        
                        # if prompt_idx == str(1):
                        #     continue

                        uc = None
                            
                        uc = model.get_learned_conditioning(n_samples * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                            
                        c = [special_embedding(model, prompt) for prompt in prompts]
                        c = torch.cat(c)
                        
                        def _build_causal_attention_mask(bsz, seq_len):
                            # lazily create causal attention mask, with full attention between the vision tokens
                            # pytorch uses additive attention mask; fill with -inf
                            mask = torch.empty(bsz, seq_len, seq_len)
                            mask.fill_(float("-inf"))
                            mask.triu_(1)  # zero out the lower diagonal
                            mask = mask.unsqueeze(1)  # expand mask
                            return mask
                        
                        causal_attention_mask = _build_causal_attention_mask(*c.shape[:2])
                        encoder = model.cond_stage_model.transformer.text_model.encoder
                        c = encoder(inputs_embeds=c, causal_attention_mask=causal_attention_mask.cuda())[0]
                        final_layer_norm = model.cond_stage_model.transformer.text_model.final_layer_norm
                        c = final_layer_norm(c)
                        # c = model.get_learned_conditioning(prompts)``
                        
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                            conditioning=c,
                                                            batch_size=n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=ddim_eta,
                                                            x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        
                        
                        for out_img in x_checked_image_torch:
                            output_path = os.path.join(
                                                        output_root_dir,
                                                        str(idx),
                                                        str(prompt_idx),
                                                    f"source{idx}_prompt{prompt_idx}_{noted[f'source{idx}_prompt{prompt_idx}']}.png")
                            
                            noted[f"source{idx}_prompt{prompt_idx}"] += 1
                            if noted[f"source{idx}_prompt{prompt_idx}"] > total_gen:
                                break
                            
                            out_img = out_img.float()
                            save_image(out_img, output_path)

# %%