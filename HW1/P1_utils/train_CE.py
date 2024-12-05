from train_utils import *

"""### Setting C"""
print("\n\nSetting C\n\n")
resnet = torchvision.models.resnet50(pretrained=False)
resnet.load_state_dict(torch.load("./pretrained_200.pt.backup"))
resnet = Model(resnet).cuda()

train(resnet, epochs, lr, "setting_c", True)

"""### Setting E"""
print("\n\nSetting E\n\n")
resnet = torchvision.models.resnet50(pretrained=False)
resnet.load_state_dict(torch.load("./pretrained_200.pt.backup"))
resnet = Model(resnet).cuda()

for param in resnet.parameters():
    param.requires_grad = False

for param in resnet.fc.parameters():
    param.requires_grad = True

train(resnet, epochs, lr, "setting_e", True)
