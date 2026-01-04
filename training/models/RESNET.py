import torch
from torch import nn

# For this RESNET the main aim is image transformation so the information needs to be preserved
class ResidualBlock(nn.Module):
    """Residual block with optional dilation for larger receptive field"""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=3, 
            stride=1, 
            padding=dilation, 
            dilation=dilation,
            bias=False
        )
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=3, 
            stride=1, 
            padding=dilation, 
            dilation=dilation,
            bias=False
        )
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


class ResNetExtractor(nn.Module):
    """
    Improved ResNet-based watermark extractor with:
    - Skip connections between encoder and decoder
    - InstanceNorm instead of BatchNorm
    - Dilated residual blocks for large receptive field
    - Minimal downsampling to preserve spatial information
    """
    def __init__(self, image_channels=3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Encoder with minimal downsampling
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        
        # Single downsample
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)
        
        # Dilated residual blocks for large receptive field without losing resolution
        self.res1 = ResidualBlock(128, dilation=2)
        self.res2 = ResidualBlock(128, dilation=2)
        self.res3 = ResidualBlock(128, dilation=2)
        self.res4 = ResidualBlock(128, dilation=2)
        self.res5 = ResidualBlock(128, dilation=4)
        self.res6 = ResidualBlock(128, dilation=4)
        self.res7 = ResidualBlock(128, dilation=4)
        self.res8 = ResidualBlock(128, dilation=4)
        self.res9 = ResidualBlock(128, dilation=1)
        
        # Decoder - upsample back to original resolution
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        
        self.deconv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.InstanceNorm2d(128, affine=True)
        
        # Final output
        self.deconv1 = nn.Conv2d(128, image_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        
        # Residual blocks with dilated convolutions
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        
        # Decoder
        x = self.relu(self.norm4(self.deconv3(x)))
        x = self.relu(self.norm5(self.deconv2(x)))
        secret_hat = torch.sigmoid(self.deconv1(x))
        
        return secret_hat
