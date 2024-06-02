import torch.nn as nn
import torch


# 双卷积结构
class DoubleCov(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doublecov = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.doublecov(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义卷积过程
        self.down_double_cov1 = DoubleCov(1, 64)
        self.down_double_cov2 = DoubleCov(64, 128)
        self.down_double_cov3 = DoubleCov(128, 256)
        self.down_double_cov4 = DoubleCov(256, 512)
        self.down_double_cov5 = DoubleCov(512, 1024)

        self.up_double_cov5 = DoubleCov(1024, 512)
        self.up_double_cov4 = DoubleCov(512, 256)
        self.up_double_cov3 = DoubleCov(256, 128)
        self.up_double_cov2 = DoubleCov(128, 64)
        # self.up_double_cov1 = DoubleCov(64, 1)

        # 定义池化层
        self.maxpool = nn.MaxPool2d(2)

        # 定义上采样层
        self.upsample4 = nn.ConvTranspose2d(1024, 512, 3, 2, 1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)
        self.upsample1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)

        # 定义输出层
        self.outcov = OutConv(64,1)

    def forward(self, x):
        # Encode
        cov0 = self.down_double_cov1(x)
        x = self.maxpool(cov0)

        cov1 = self.down_double_cov2(x)
        x = self.maxpool(cov1)

        cov2 = self.down_double_cov3(x)
        x = self.maxpool(cov2)

        cov3 = self.down_double_cov4(x)
        x = self.maxpool(cov3)

        x = self.down_double_cov5(x)

        # Decode
        up_cov3 = self.upsample4(x)  # 512×64×64
        x = torch.cat([up_cov3, cov3], dim=1)  # 1024×64×64
        x = self.up_double_cov5(x)  # 512×64×64

        up_cov2 = self.upsample3(x)  # 256×128×128
        x = torch.cat([up_cov2, cov2], dim=1)  # 512×128×128
        x = self.up_double_cov4(x)  # 256×128×128

        up_cov1 = self.upsample2(x)  # 128×256×256
        x = torch.cat([up_cov1, cov1], dim=1)  # 256×256×256
        x = self.up_double_cov3(x)  # 128×256×256

        up_cov0 = self.upsample1(x)  # 64×512×512
        x = torch.cat([up_cov0, cov0], dim=1)  # 128×512×512
        x = self.up_double_cov2(x)  # 64×512×512

        # output = self.up_double_cov1(x)  # 1×512×512
        output = self.outcov(x)
        return output
