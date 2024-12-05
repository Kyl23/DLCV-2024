# %%
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.chdir('../')

# %%
os.getcwd()

# %%
IMAGE_SIZE = (512, 512)

# %%
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# %%
import torch

# %%
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
config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "sd-v1-4.ckpt")

# %%
from torch import nn

embedding_layer = model.cond_stage_model.transformer.text_model.embeddings.token_embedding
position_encoding_layer = model.cond_stage_model.transformer.text_model.embeddings.position_embedding
tokenizer = model.cond_stage_model.tokenizer
encoder = model.cond_stage_model.transformer.text_model.encoder
final_layer_norm = model.cond_stage_model.transformer.text_model.final_layer_norm

string_to_token_dict = {}
string_to_param_dict = nn.ParameterDict()
        
vocab_size, embedding_dim = embedding_layer.weight.shape
original_weights = embedding_layer.weight.data

new_tokens = ['<new1>', '<new2>']
# token_params = torch.nn.Parameter(torch.rand(size=(len(new_tokens), original_weights.size(1)), requires_grad=True))

def get_initial_token_with_promp(prompt):
    if prompt == "" or prompt == None:
        return nn.Parameter(torch.rand(1, original_weights.size(1)))
    
    token = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']

    token = token.cuda()
    return nn.Parameter(embedding_layer(token).squeeze(0))

token_map_prompt = ["yellow dog", "fantastical"]
for i, new_token in enumerate(new_tokens):
    string_to_token_dict[new_token] = tokenizer(new_token)
    string_to_param_dict[new_token] = get_initial_token_with_promp(token_map_prompt[i]) # token_params[i]

# %%
import re

def split_text_by_token(text):
    pattern = r"(<new1>|<new2>)"
    split_text = re.split(pattern, text)
    result = [item for item in split_text if item]
    
    return result

def special_embedding(text):
    device = model.device 
    max_length = 77
    bos, eos = tokenizer.bos_token, tokenizer.eos_token
    bos = tokenizer(bos, return_tensors='pt', add_special_tokens=False).to(device)['input_ids']
    eos = tokenizer(eos, return_tensors='pt', add_special_tokens=False).to(device)['input_ids']
    
    embedded_text = [embedding_layer(bos)]
    
    text_list = split_text_by_token(text)
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
    
# %%
for p in string_to_param_dict.parameters():
    p.requires_grad = True
    
model.logvar = model.logvar.cuda()


model = model.train()
# %%
from tqdm import tqdm
from torch import optim
from torch.amp import autocast, GradScaler
from contextlib import nullcontext


dataloader = None
def train(start, epoch, train_tokens=[], pretrained=None, out_dir="train_model", learning_rate=5e-3, fp16=True):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    global string_to_param_dict
    if pretrained is not None:
        string_to_param_dict = torch.load(pretrained)

    opt = optim.AdamW([string_to_param_dict[train_token] for train_token in train_tokens], lr = learning_rate)
    scaler = GradScaler()

    context = autocast if fp16 else nullcontext
    with context('cuda'):
        for i in range(start, start + epoch):
            p_bar = tqdm(dataloader)
            p_bar.set_description(f"ecpoh {i+1}/{start + epoch}: ")
            total_loss = 0
            count = 0
            for batch in p_bar:
                x = batch['img']
                c = batch['caption']

                x = x.cuda()

                c = ([special_embedding(_c) for _c in c])
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
                c = encoder(inputs_embeds=c, causal_attention_mask=causal_attention_mask.cuda())[0]
                c = final_layer_norm(c)

                encoder_posterior = model.encode_first_stage(x)
                z = model.get_first_stage_encoding(encoder_posterior).detach()

                opt.zero_grad()
                model.zero_grad()

                loss = model(z, c)[0]

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                count += 1
                total_loss += loss.item()
                p_bar.set_postfix(loss = f"{total_loss / count:.2f}")
                
                
            torch.save(string_to_param_dict, f"{out_dir}/{i}.ckpt")

# %%
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T

tf = T.Compose([
            T.ToTensor(),
         ])


# imagenet_templates_small = [
#     "a photo of a {}",
#     "a photo of my {}",
#     "a photo of a small {}"
# ]

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    # "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    # "a bright photo of the {}",
    # "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    # "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    # "a photo of the weird {}",
    # "a photo of the large {}",
    # "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

# imagenet_style_templates_small = [
#     "A girl running through a forest in manga style by {}",
#     "A lush forest scene in manga style by {}",
#     "A girl sprinting among the trees in manga art by {}",
#     "Manga-style depiction of an island by {}",
#     "An energetic girl in a forest, manga style by {}"
# ]

class ImgDataset(Dataset):
    def __init__(self, root_dirs=[
                                    "../hw2_data/textual_inversion/0", 
                                  "../hw2_data/textual_inversion/1"
    ], label="../hw2_data/textual_inversion/input.json"):
        self.path = []
        self.label = []
        
        import json
        with open(label) as f:
            labels = json.load(f)

        for i, root_dir in  enumerate(root_dirs):
            i = os.path.basename(root_dir)
            for p in os.listdir(root_dir):
                c = labels[str(i)]['token_name']

                # self.path.append(os.path.join(root_dir, p))
                # self.label.append(c) 
                for template in imagenet_templates_small if i == str(0) else imagenet_style_templates_small:
                    self.path.append(os.path.join(root_dir, p))
                    self.label.append(template.format(c)) 

    def read_image(self,path):
        img = Image.open(path)
        img = img.resize(IMAGE_SIZE)

        return img  

    # working for indexing
    def __getitem__(self, index):
        return {'img': tf(self.read_image(self.path[index])), 'caption': self.label[index]}

    # return the length of our dataset
    def __len__(self):
        return len(self.path)

# %%
'''
Train style image
'''
dataloader = DataLoader(ImgDataset(root_dirs=["../textual_inversion/1"]), batch_size=4, shuffle=True, num_workers=8, persistent_workers=True)

train(start=0, epoch=10, train_tokens=["<new2>"], pretrained=None, out_dir='train_model') # the best 3.ckpt

# %%
'''
Train dog image
'''
dataloader = DataLoader(ImgDataset(root_dirs=["../hw2_data/textual_inversion/0"]), batch_size=4, shuffle=True, num_workers=8, persistent_workers=True)
train(start=0, epoch=30, train_tokens=["<new1>"], pretrained='train_model/3.ckpt', out_dir='train_model1') # the best 16.ckpt

# %%
'''
hw2_3
'''
class ImgDataset(Dataset):
    def __init__(self, root_dirs=[
                                    "../hw2_data/textual_inversion/0", 
                                  "../hw2_data/textual_inversion/1"
    ], label="../hw2_data/textual_inversion/input.json"):
        self.path = []
        self.label = []
        
        import json
        with open(label) as f:
            labels = json.load(f)

        for i, root_dir in  enumerate(root_dirs):
            i = os.path.basename(root_dir)
            for p in os.listdir(root_dir):
                try:
                    c = labels[str(i)]['token_name']
                except:
                    c = '<new2>'
                # self.path.append(os.path.join(root_dir, p))
                # self.label.append(c) 
                for template in imagenet_templates_small:
                    self.path.append(os.path.join(root_dir, p))
                    self.label.append(template.format(c)) 

    def read_image(self,path):
        img = Image.open(path)
        img = img.resize(IMAGE_SIZE)

        return img  

    # working for indexing
    def __getitem__(self, index):
        return {'img': tf(self.read_image(self.path[index])), 'caption': self.label[index]}

    # return the length of our dataset
    def __len__(self):
        return len(self.path)

token_map_prompt = ["yellow dog", "cat"] 
for i, new_token in enumerate(new_tokens):
    string_to_token_dict[new_token] = tokenizer(new_token)
    string_to_param_dict[new_token] = get_initial_token_with_promp(token_map_prompt[i]) # token_params[i]
dataloader = DataLoader(ImgDataset(root_dirs=["../hw2_data/textual_inversion/0", "../hw2_data/textual_inversion/cat"]), batch_size=4, shuffle=True, num_workers=8, persistent_workers=True)
train(start=0, epoch=10, train_tokens=["<new1>", "<new2>"], pretrained=None, out_dir='train_model_dog_cat', learning_rate=5e-3, fp16=False)

# %%
