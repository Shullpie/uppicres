import torch
import torch.nn as nn

from model.modules.optim import activation_funcs


class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation_function
                 ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            activation_function,

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            activation_function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Unet256(nn.Module):
    name = 'unet256'
    
    def __init__(self, options: dict):
        super().__init__()

        self.in_channels = options.get('in_channels', None)
        self.out_channels = options.get('out_channels', None)
        self.activation_func = activation_funcs.get_activation_function(options.get('activation_function', None))

        self.maxpool = nn.MaxPool2d(2, 2)

        self.down_0 = DoubleConv(self.in_channels, 64, self.activation_func)
        self.down_1 = DoubleConv(64, 128, self.activation_func)
        self.down_2 = DoubleConv(128, 256, self.activation_func)
        self.down_3 = DoubleConv(256, 512, self.activation_func)

        self.bottleneck = DoubleConv(512, 1024, self.activation_func)

        self.up_31 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_32 = DoubleConv(1024, 512, self.activation_func)
        self.up_21 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_22 = DoubleConv(512, 256, self.activation_func)
        self.up_11 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_12 = DoubleConv(256, 64, self.activation_func)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),

            self.activation_func,

            nn.Conv2d(
                in_channels=64,
                out_channels=self.out_channels,
                kernel_size=1,
            )
        )

    def forward(self, x: torch.Tensor):
        # 3x256x256 -> 64x256x256
        conv0 = self.down_0(x)

        # 64x256x256 -> 128x256x256 -> 128x128x128
        conv1 = self.down_1(conv0)
        x = self.maxpool(conv1)

        # 128x128x128 -> 256x128x128 -> 256x64x64
        conv2 = self.down_2(x)
        x = self.maxpool(conv2)

        # 256x64x64 -> 512x64x64 -> 512x32x32
        conv3 = self.down_3(x)
        x = self.maxpool(conv3)

        # 512x32x32 -> 1024x32x32
        x = self.bottleneck(x)

        # 1024x32x32 -> 512x64x64 ->1024x64x64 -> 512x64x64
        x = self.up_31(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_32(x)

        # 512x64x64 -> 256x128x128 -> 512x128x128 -> 256x128x128
        x = self.up_21(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_22(x)

        # 256x128x128 -> 128x256x256 -> 256x256x256 -> 64x256x256
        x = self.up_11(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_12(x)

        # 64x256x256 -> 128x256x256
        x = torch.cat([x, conv0], dim=1)

        # 128x256x256 -> 64x256x256 ->1x256x256
        x = self.final_conv(x)
        return x
