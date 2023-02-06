
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 3), nn.BatchNorm2d(512), nn.GELU(),
            nn.Conv2d(512, 1024, 3), nn.BatchNorm2d(1024), nn.GELU(),
            nn.Conv2d(1024, 6, 1), nn.BatchNorm2d(6), nn.GELU()
        )
        self.fc = nn.Sequential(nn.Linear(6 * 4 * 4, 256), nn.GELU(), nn.Linear(256, 16), nn.GELU())
        self.last = nn.Linear(16, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.last(x)

        return x


net = Net()