import torch.nn as nn
import torch.nn.functional as F
import torch

device = 'cuda'

class BaselineModel(torch.nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()

        # Convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(12288, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x, proposals):
        # Pass input through 3D convolutional layers
        x = self.conv_layers(x)

        x = x.reshape(x.shape[0], -1)

        # Pass input through fully connected layers
        x = self.fc_layers(x)

        x = F.normalize(x, p=2, dim=1)  # L2-normalize output embeddings

        return x
