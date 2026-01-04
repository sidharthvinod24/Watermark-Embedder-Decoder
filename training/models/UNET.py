import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, alpha=0.05):
        super().__init__()
        self.alpha = alpha

        self.encoder1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, cover, x):
        enc1 = self.encoder1(x)
        enc1_pool = self.pool1(enc1)

        enc2 = self.encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2)

        bottleneck = self.bottleneck(enc2_pool)

        dec1 = self.upconv1(bottleneck)
        # Concatenate the encoder output with the decoder output (skip connection)
        dec1 = self.decoder1(torch.cat([dec1, enc2], dim=1))

        dec2 = self.upconv2(dec1)
        dec2 = self.decoder2(torch.cat([dec2, enc1], dim=1))
        residual = torch.tanh(self.final_conv(dec2))
        stego = torch.clamp(cover + self.alpha * residual, 0.0, 1.0)
        return stego
