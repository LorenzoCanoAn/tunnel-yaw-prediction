import torch
from torch import nn
from torchsummary import summary


class YawEstimator(nn.Module):
    def __init__(self):
        self.input_height = 30
        self.input_width = 30
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, [3, 3], padding=(1, 1), padding_mode="zeros"),
            nn.Dropout(p=0.05),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Conv2d(8, 16, [3, 3], padding=(1, 1), padding_mode="zeros"),
            nn.Dropout(p=0.05),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Conv2d(16, 32, [3, 3], padding=(1, 1), padding_mode="zeros"),
            nn.Dropout(p=0.05),
            nn.MaxPool2d([2, 2]),
            nn.ReLU(),
            nn.Conv2d(32, 64, [3, 3], padding=(1, 1), padding_mode="zeros"),
            nn.Dropout(p=0.05),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits


if __name__ == "__main__":
    print(summary(YawEstimator(), (1, 30, 30), device="cpu"))
