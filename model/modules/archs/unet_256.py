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
    name ='unet256'
    def __init__(self, options: dict):
        super().__init__()

        options = options['nns']['models'][self.name]
        self.in_channels = options.get('in_channels', None)
        self.out_channels = options.get('out_channels', None)
        print(self.in_channels, self.out_channels)

        self.activation_func = activation_funcs\
                              .get_activation_function(options.get('activation_function', None))
        print(self.activation_func)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down_0 = DoubleConv(self.in_channels, 64, self.activation_func)
        self.down_1 = DoubleConv(64, 128, self.activation_func)
        self.down_2 = DoubleConv(128, 256, self.activation_func)
        self.down_3 = DoubleConv(256, 512, self.activation_func)

        self.bottleneck = DoubleConv(512, 1024, self.activation_func)

        self.up_3 = DoubleConv(1024+512, 512, self.activation_func)
        self.up_2 = DoubleConv(512+256, 256, self.activation_func)
        self.up_1 = DoubleConv(256+128, 128, self.activation_func)

        self.final_conv1 = DoubleConv(128, 64, self.activation_func)
        self.final_conv2 = nn.Conv2d(64, self.out_channels, 1)

    def forward(self, x: torch.Tensor):
        # 3x256x256 -> 64x256x256
        x = self.down_0(x)

        # 64x256x256 -> 128x256x256 -> 128x128x128
        conv1 = self.down_1(x)
        x = self.maxpool(conv1)

        # 128x128x128 -> 256x128x128 -> 256x64x64
        conv2 = self.down_2(x)
        x = self.maxpool(conv2)

        # 256x64x64 -> 512x64x64 -> 512x32x32
        conv3 = self.down_3(x)
        x = self.maxpool(conv3)

        # 512x32x32 -> 1024x32x32
        x = self.bottleneck(x)

        # 1024x32x32 -> 1024x64x64 ->1024+512x64x64 -> 512x64x64
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_3(x)

        # 512x64x64 -> 512x128x128 -> 512+256x128x128 -> 256x128x128
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_2(x)

        # 256x128x128 -> 256x256x256 -> 256+128x256x256 -> 128x256x256
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_1(x)

        # 128x256x256 -> 64x256x256 -> 1x256x256
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x
