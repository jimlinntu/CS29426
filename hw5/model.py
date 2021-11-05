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

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_keypoints = 58
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, padding=1)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)

        # FC layer
        self.fc = nn.Linear(256*5*7, 256)
        self.fc2 = nn.Linear(256, self.num_keypoints*2)

    def forward(self, img):
        # (1, 180, 240) -> (16, 90, 120)
        x = self.conv1(img)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (16, 90, 120) -> (32, 45, 60)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (32, 45, 60) -> (64, 22, 30)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (64, 22, 30) -> (128, 11, 15)
        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (256, 11, 15) -> (256, 5, 7)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (128, 11, 15) -> (n_keypoints, 2)
        batch_size = x.shape[0]
        out = self.fc2(self.relu(self.fc(x.reshape(batch_size, -1))))
        out = out.reshape(batch_size, self.num_keypoints, 2)
        return out

class Baseline_5x5(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_keypoints = 58
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(1, 16, 5, 1, padding=2)

        self.conv2 = nn.Conv2d(16, 32, 5, 1, padding=2)

        self.conv3 = nn.Conv2d(32, 64, 5, 1, padding=2)

        self.conv4 = nn.Conv2d(64, 128, 5, 1, padding=2)

        self.conv5 = nn.Conv2d(128, 256, 5, 1, padding=2)

        # FC layer
        self.fc = nn.Linear(256*5*7, 256)
        self.fc2 = nn.Linear(256, self.num_keypoints*2)

    def forward(self, img):
        # (1, 180, 240) -> (16, 90, 120)
        x = self.conv1(img)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (16, 90, 120) -> (32, 45, 60)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (32, 45, 60) -> (64, 22, 30)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (64, 22, 30) -> (128, 11, 15)
        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (256, 11, 15) -> (256, 5, 7)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (128, 11, 15) -> (n_keypoints, 2)
        batch_size = x.shape[0]
        out = self.fc2(self.relu(self.fc(x.reshape(batch_size, -1))))
        out = out.reshape(batch_size, self.num_keypoints, 2)
        return out

class Baseline_7x7(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_keypoints = 58
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(1, 16, 7, 1, padding=3)

        self.conv2 = nn.Conv2d(16, 32, 7, 1, padding=3)

        self.conv3 = nn.Conv2d(32, 64, 7, 1, padding=3)

        self.conv4 = nn.Conv2d(64, 128, 7, 1, padding=3)

        self.conv5 = nn.Conv2d(128, 256, 7, 1, padding=3)

        # FC layer
        self.fc = nn.Linear(256*5*7, 256)
        self.fc2 = nn.Linear(256, self.num_keypoints*2)

    def forward(self, img):
        # (1, 180, 240) -> (16, 90, 120)
        x = self.conv1(img)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (16, 90, 120) -> (32, 45, 60)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (32, 45, 60) -> (64, 22, 30)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (64, 22, 30) -> (128, 11, 15)
        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (256, 11, 15) -> (256, 5, 7)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (128, 11, 15) -> (n_keypoints, 2)
        batch_size = x.shape[0]
        out = self.fc2(self.relu(self.fc(x.reshape(batch_size, -1))))
        out = out.reshape(batch_size, self.num_keypoints, 2)
        return out

class Baseline_9x9(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_keypoints = 58
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(1, 16, 9, 1, padding=4)

        self.conv2 = nn.Conv2d(16, 32, 9, 1, padding=4)

        self.conv3 = nn.Conv2d(32, 64, 9, 1, padding=4)

        self.conv4 = nn.Conv2d(64, 128, 9, 1, padding=4)

        self.conv5 = nn.Conv2d(128, 256, 9, 1, padding=4)

        # FC layer
        self.fc = nn.Linear(256*5*7, 256)
        self.fc2 = nn.Linear(256, self.num_keypoints*2)

    def forward(self, img):
        # (1, 180, 240) -> (16, 90, 120)
        x = self.conv1(img)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (16, 90, 120) -> (32, 45, 60)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (32, 45, 60) -> (64, 22, 30)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (64, 22, 30) -> (128, 11, 15)
        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (256, 11, 15) -> (256, 5, 7)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # (128, 11, 15) -> (n_keypoints, 2)
        batch_size = x.shape[0]
        out = self.fc2(self.relu(self.fc(x.reshape(batch_size, -1))))
        out = out.reshape(batch_size, self.num_keypoints, 2)
        return out
