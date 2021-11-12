import torch

from torch import nn

from resnet import resnet18, resnet34

import segmentation_models_pytorch as smp

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True, progress=True)

        old_fc = self.resnet.fc
        last = nn.Sequential(
                nn.Linear(old_fc.in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 68*2))

        # Overwrite the last layer
        self.resnet.fc = last

    def forward(self, x):
        out = self.resnet(x)
        out = out.reshape(-1, 68, 2)
        return out

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True, progress=True)

        old_fc = self.resnet.fc
        self.fc = nn.Sequential(
                nn.Linear(old_fc.in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 68*2))

        # Overwrite the last layer
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        out = self.resnet(x)
        out = self.fc(out)
        out = out.reshape(-1, 68, 2)
        return out

class Model_res34(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(pretrained=True, progress=True)

        old_fc = self.resnet.fc
        self.fc = nn.Sequential(
                nn.Linear(old_fc.in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 68*2))

        # Overwrite the last layer
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        out = self.resnet(x)
        out = self.fc(out)
        out = out.reshape(-1, 68, 2)
        return out

class ModelUnet(nn.Module):
    # https://github.com/qubvel/segmentation_models.pytorch#installation
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=68)

    def forward(self, x):
        return self.unet(x)
