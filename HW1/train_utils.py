import warnings
warnings.filterwarnings("ignore")

from torch import nn

num_classes = 65

class Model(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        num_features = backbone.fc.out_features

        self.fc = nn.Sequential(
#            nn.ReLU(),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        return x

