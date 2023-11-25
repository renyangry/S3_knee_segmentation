import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import *
from collections import OrderedDict

class UNet2D(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet2D, self).__init__()

        features = init_features
        #self.droplayer = nn.Dropout(0.15)
        self.encoder1 = UNet2D._block(in_channels, features, name="enc1")
        self.pool1 = (nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder2 = UNet2D._block(features, features * 2, name="enc2")
        self.pool2 = (nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder3 = UNet2D._block(features * 2, features * 4, name="enc3")
        self.pool3 = (nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder4 = UNet2D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
#        for 2.5D U-Net
#        if x.shape[1] == 1:
#            x=x.repeat(1,3,1,1,1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2((self.pool1(enc1))) 
        enc3 = self.encoder3((self.pool2(enc2)))
        enc4 = self.encoder4((self.pool3(enc3)))
        
#        enc2 = self.encoder2(self.droplayer(self.pool1(enc1))) # 0.25 changed to 0.5 
#        enc3 = self.encoder3(self.droplayer(self.pool2(enc2)))
#        enc4 = self.encoder4(self.droplayer(self.pool3(enc3)))
        
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1) #torch.sigmoid(self.conv(dec1))


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),   # InstanceNorm2d   LeakyReLU
                    (name + "norm1", nn.InstanceNorm2d(num_features=features)), # accelerate the network convergence using InstanceNorm2d
                    (name + "relu1", nn.LeakyReLU(inplace=True)), # ReLU
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(inplace=True)),
                ]
            )
        )


class Unet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(Unet3D, self).__init__()
        features = init_features
        self.down1 = self._block(in_channels, features, name="down1")
        self.down2 = self._block(features, features * 2, name="down2")
        self.down3 = self._block(features * 2, features * 4, name="down3")
        self.down4 = self._block(features * 4, features * 8, name="down4")
        self.conv1 = nn.Conv3d(features * 8, features * 16, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(features * 16)
        self.conv2 = nn.Conv3d(features * 16, features * 16, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(features * 16)
        self.up4 = self._block(features * 16, features * 8, name="up4")
        self.up3 = self._block(features * 8, features * 4, name="up3")
        self.up2 = self._block(features * 4, features * 2, name="up2")
        self.up1 = self._block(features * 2, features, name="up1")
        self.outconv = nn.Conv3d(features, out_channels, 1)

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.dropout3d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout3d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x = self.outconv(x)

        return x

    @staticmethod
    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv3d(in_channels, features, 3, padding=1)),
                    (name + "bn1", nn.BatchNorm3d(features)),
                    (name + "conv2", nn.Conv3d(features, features, 3, padding=1)),
                    (name + "bn2", nn.BatchNorm3d(features)),
                ]
            )
        )
