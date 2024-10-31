import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        print(f"x1 shape: {x1.shape}")  # 输出形状
        x2 = self.conv2(x1)
        print(f"x2 shape: {x2.shape}")  # 输出形状
        x3 = self.conv3(x2)
        print(f"x3 shape: {x3.shape}")  # 输出形状
        x4 = self.conv4(x3)
        print(f"x4 shape: {x4.shape}")  # 输出形状

        # Decoder forward pass
        x5 = self.deconv1(x4)
        print(f"x5 shape: {x5.shape}")  # 输出形状
        x6 = self.deconv2(x5)
        print(f"x6 shape: {x6.shape}")  # 输出形状
        x7 = self.deconv3(x6)
        print(f"x7 shape: {x7.shape}")  # 输出形状
        output = self.deconv4(x7)
        print(f"output shape: {output.shape}")  # 输出形状

        return output
