from typing import Optional, Callable, Literal

import torch
import torch.nn as nn

from ..utils import activation_funcs


class BatchDoubleConv(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 activation_function):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1,
                      bias=False),

            nn.BatchNorm2d(out_channels),

            activation_function,

            nn.Conv2d(in_channels=out_channels, 
                      out_channels=out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1,
                      bias=False),
            
            nn.BatchNorm2d(out_channels),

            activation_function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SegUnet(nn.Module):
    name = "SegUnet"

    def __init__(self,
                 img_size: Literal[256] | Literal[512] | Literal[1024],
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 activation_function: Optional[Callable] = None) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_function = activation_function
        self.img_size = img_size if img_size else 1024
        self.features = self._fill_features(img_size=self.img_size)
        
        if (self.in_channels is not None 
            and self.out_channels is not None 
            and self.activation_function_func is not None):
            self._init_parts()

    def init_from_config(self, nn_options: dict) -> None:
        if self.in_channels is None:
            self.in_channels = nn_options["in_channels"]
        if self.out_channels is None:
            self.out_channels = nn_options["out_channels"]
        if self.activation_function is None:
            self.activation_function = activation_funcs.get_activation_function(nn_options["activation_function"])
        self._init_parts()

    def forward(self, x: torch.Tensor):
        skip_connections = []

        # Down
        # crop 256: 3x256x256 -> 64x128x128 -> 128x64x64 -> 256x32x32
        # crop 512: 3x512x512 -> 64x256x256 -> 128x128x128 -> 256x64x64 -> 512x32x32
        # crop 0/1024: 3x1024x1024 -> 64x512x512 -> 128x256x256 -> 256x128x128 -> 512x64x64 -> 1024x32x32
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        # crop 256: 256x32x32 -> 512x32x32   
        # crop 512: 512x32x32 -> 1024x32x32
        # crop 0/1024: 1024x32x32 -> 2048x32x32
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Up
        # crop 256: 512x32x32 -> 256x64x64 -> 128x128x128 -> 64x256x256
        # crop 512: 1024x32x32 -> 512x64x64 -> 256x128x128 -> 128x256x256 -> 64x512x512
        # crop 0/1024: 2048x32x32 -> 1024x64x64 -> 512x128x128 -> 256x256x256 -> 128x512x512 -> 64x1024x1024 
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            x = cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        
        # Final
        # crop 256: 64x256x256 -> 1x256x256
        # crop 512: 64x512x512 -> 1x512x512
        # crop 0/1024: 64x1024x1024 -> 1x1024x1024
        return self.final_conv(x)
    
    @staticmethod
    def _fill_features(img_size: int) -> tuple[int]:
        features = [64, 128, 256]

        while features[-1] < img_size:
            features.append(features[-1]*2)
        return tuple(features)

    def _init_parts(self):
        # Down part
        for feature in self.features:
            self.downs.append(BatchDoubleConv(self.in_channels, feature, self.activation_function))
            self.in_channels = feature

        # Up part    
        for feature in reversed(self.features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(BatchDoubleConv(feature*2, feature, self.activation_function))

        # Bottleneck   
        self.bottleneck = BatchDoubleConv(self.features[-1], 2*self.features[-1], self.activation_function)

        # Final
        self.final_conv = BatchDoubleConv(self.features[0], self.out_channels, self.activation_function)
