from train_utils import *

"""### Setting A"""
print("\n\nSetting A\n\n")
resnet = torchvision.models.resnet50(pretrained=False)
resnet = Model(resnet).cuda()

train(resnet, epochs, lr, "setting_a", True)

"""### Setting B"""
print("\n\nSetting B\n\n")
resnet = torchvision.models.resnet50(pretrained=False)
resnet.load_state_dict(torch.load("./p1_data/pretrain_model_SL.pt"))
resnet = Model(resnet).cuda()

train(resnet, epochs, lr, "setting_b", True)

"""### Setting D"""
print("\n\nSetting D\n\n")
resnet = torchvision.models.resnet50(pretrained=False)
resnet.load_state_dict(torch.load("./p1_data/pretrain_model_SL.pt"))
resnet = Model(resnet).cuda()

for param in resnet.parameters():
    param.requires_grad = False

for param in resnet.fc.parameters():
    param.requires_grad = True

train(resnet, epochs, lr, "setting_d", True)
