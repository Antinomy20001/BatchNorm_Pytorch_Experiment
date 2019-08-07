import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from BatchNorm import BatchNorm


def weight_init(m, scale):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=scale)
        torch.nn.init.normal_(m.bias, std=scale)


class CNN(nn.Module):
    def __init__(self, init_scale, activ_func):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.activ1 = activ_func()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.activ2 = activ_func()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        for layer in self.modules():
            weight_init(layer, init_scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.activ2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CNN_BN(nn.Module):
    def __init__(self, init_scale, activ_func):
        super(CNN_BN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.bn1 = BatchNorm(6)
        self.activ1 = activ_func()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = BatchNorm(16)
        self.activ2 = activ_func()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = BatchNorm(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = BatchNorm(84)
        self.fc3 = nn.Linear(84, 10)
        self.bn5 = BatchNorm(10)

        for layer in self.modules():
            weight_init(layer, init_scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activ2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.bn4(x)
        x = self.fc3(x)
        x = self.bn5(x)
        return x
