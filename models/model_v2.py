import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModelV2(nn.Module):
    def __init__(self):
        super(CNNModelV2, self).__init__()

        # ----- FEATURE EXTRACTION BLOCK -----
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # ----- CLASSIFIER BLOCK -----
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)   # 2 classes: cat, dog

    def forward(self, x):

        # BLOCK 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 128 → 64

        # BLOCK 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 64 → 32

        # BLOCK 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 32 → 16

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Helper function (optional but recommended)
def get_model_v2():
    return CNNModelV2()