from typing import Optional, Callable, Literal

import torch
import torch.nn as nn
from torchvision.transforms.functional import hflip

from model.modules.optim import activation_funcs
from model.data.processing import functional
from utils.types import MaskTorch


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


class UnetWide(nn.Module):
    name = "unetwide"

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
        self.n_features = self._fill_n_features(img_size=self.img_size)

        if (self.in_channels is not None
            and self.out_channels is not None
            and self.activation_function is not None):
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
        x = self.downs[0](x)
        for down in self.downs[1:]:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        # crop 512: 1024x32x32 -> 2048x32x32
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        # Up
        # crop 512: 2048x32x32 -> 1024x32x32 -> 512x64x64 -> 256x128x128 -> 128x256x256 -> 64x512x512
        for idx in range(0, len(self.ups)-2, 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        x = self.ups[-1](x)
        # Final
        # crop 512: 64x512x512 -> 1x512x512

        return self.final_conv(x)

    @staticmethod
    def _fill_n_features(img_size: int) -> tuple[int]:
        n_features = [64, 128, 256]

        while n_features[-1] < img_size*2:
            n_features.append(n_features[-1]*2)
        return tuple(n_features)

    def _init_parts(self):
        if isinstance(self.activation_function, str):
            self.activation_function = activation_funcs.get_activation_function(self.activation_function)
        # Down part
        for f in self.n_features:
            self.downs.append(BatchDoubleConv(self.in_channels, f, self.activation_function))
            self.in_channels = f

        # Up part
        for f in reversed(self.n_features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.ups.append(BatchDoubleConv(f*2, f, self.activation_function))

        # Bottleneck
        self.bottleneck = BatchDoubleConv(
            in_channels=self.n_features[-1], 
            out_channels=2*self.n_features[-1], 
            activation_function=self.activation_function
        )

        # Final
        self.final_conv = nn.Conv2d(
            self.n_features[0],
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
    def inference(
            self, 
            user_img: torch.Tensor, 
            inference_options: dict, 
            device: str
    ) -> MaskTorch:
        self.eval()
        res = []
        n_patches = len(user_img)

        options = inference_options['seg']
        crop = inference_options.get('crop', None)
        step = inference_options.get('patches_to_device', 2)
        pos_label_threshold = options.get('pos_label_threshold', 0.5)

        mean = options.get('normalize', None).get('mean', None)
        std = options.get('normalize', None).get('std', None)

        flipped_image = hflip(user_img)

        user_img, dims = functional.prepare_img(
            img=user_img, 
            mean=mean,
            std=std,
            crop=crop
        )  
        flipped_image, _ = functional.prepare_img(
            img=flipped_image, 
            mean=mean,
            std=std,
            crop=crop
        )    
        
        for img in [user_img, flipped_image]:
            mask = []
            for i in range(0, n_patches, step):
                if i+step > n_patches:
                    tmp = img[i:].to(device)
                    
                    for patch in self(tmp):
                        mask.append(patch.cpu())
                    break

                tmp = img[i:i+step].to(device)
                for patch in self(tmp):
                    mask.append(patch.cpu())
            res.append(torch.stack(mask))
        mask1, mask2 = res
        del res

        mask1 = torch.sigmoid(functional.cat_patches(mask1, dims))
        mask2 = torch.sigmoid(functional.cat_patches(mask2, dims))
        mask2 = hflip(mask2)

        mask1 = mask1 > pos_label_threshold
        mask2 = mask2 > pos_label_threshold
        return torch.logical_or(mask1, mask2).to(torch.float32)   
