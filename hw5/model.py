import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)

        # FC layer
        self.fc = nn.Linear(64*7*10, 128)
        self.fc2 = nn.Linear(128, num_keypoints*2)

    def forward(self, img):
        # (1, 60, 80) -> (16, 30, 40)
        x = self.conv1(img)
        x = self.relu(x)
        x = self.maxpool(x)

        # (16, 30, 40) -> (32, 15, 20)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (32, 15, 20) -> (64, 7, 10)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (64, 7, 10) -> (n_keypoints, 2)
        batch_size = x.shape[0]
        out = self.fc2(self.relu(self.fc(x.reshape(batch_size, -1))))
        out = out.reshape(batch_size, self.num_keypoints, 2)
        return out

class SimpleModelDeeper(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)

        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)

        # FC layer
        self.fc = nn.Linear(64*7*10, 128)
        self.fc2 = nn.Linear(128, num_keypoints*2)

    def forward(self, img):
        # (1, 60, 80) -> (16, 30, 40)
        x = self.conv1(img)
        x = self.relu(x)
        x = self.maxpool(x)

        # (16, 30, 40) -> (32, 15, 20)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (32, 15, 20) -> (64, 7, 10)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)

        # (64, 7, 10) -> (n_keypoints, 2)
        batch_size = x.shape[0]
        out = self.fc2(self.relu(self.fc(x.reshape(batch_size, -1))))
        out = out.reshape(batch_size, self.num_keypoints, 2)
        return out

class SimpleModelLargeKernel(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(1, 16, 5, 1, padding=2)

        self.conv2 = nn.Conv2d(16, 32, 5, 1, padding=2)

        self.conv3 = nn.Conv2d(32, 64, 5, 1, padding=2)

        # FC layer
        self.fc = nn.Linear(64*7*10, 128)
        self.fc2 = nn.Linear(128, num_keypoints*2)

    def forward(self, img):
        # (1, 60, 80) -> (16, 30, 40)
        x = self.conv1(img)
        x = self.relu(x)
        x = self.maxpool(x)

        # (16, 30, 40) -> (32, 15, 20)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (32, 15, 20) -> (64, 7, 10)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (64, 7, 10) -> (n_keypoints, 2)
        batch_size = x.shape[0]
        out = self.fc2(self.relu(self.fc(x.reshape(batch_size, -1))))
        out = out.reshape(batch_size, self.num_keypoints, 2)
        return out
