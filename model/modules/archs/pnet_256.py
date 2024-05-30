import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.optim import activation_funcs
from model.data.processing import functional
from utils.types import MaskTorch, ImageTorch, ImagePIL


class PartialConv2d(nn.Conv2d):
    ###############################################################################
    # BSD 3-Clause License
    #
    # Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
    #
    # Author & Contact: Guilin Liu (guilinl@nvidia.com)
    ###############################################################################
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, 
                                                 self.in_channels, 
                                                 self.kernel_size[0], 
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * \
            self.weight_maskUpdater.shape[2] * \
            self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, x, mask_in=None):
        assert len(x.shape) == 4
        if mask_in is not None or self.last_size != tuple(x.shape):
            self.last_size = tuple(x.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(x.data.shape[0], x.data.shape[1], x.data.shape[2], x.data.shape[3]).to(x)
                    else:
                        mask = torch.ones(1, 1, x.data.shape[2], x.data.shape[3]).to(x)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-6)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(x, mask) if mask_in is not None else x)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PConvNet256(nn.Module):
    name = 'pnet256'

    def __init__(self, options: dict):
        super(PConvNet256, self).__init__()
        self.activation_function_1 = activation_funcs.get_activation_function(options['activation_function_1'])
        self.activation_function_2 = activation_funcs.get_activation_function(options['activation_function_2'])
        self.mean = options.get('normalize', None).get('mean')
        self.std = options.get('normalize', None).get('std')
        self.patches_to_device = options.get('patches_to_device', 4) 
        self.freeze_bn = options.get('freeze_bn', False)
        self.normalization_layer = nn.BatchNorm2d
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

        # 3x256x256 -> 64x128x128
        self.block_1 = PartialConv2d(in_channels=3, out_channels=64, kernel_size=5, 
                                     stride=2, padding=2, multi_channel=True, return_mask=True)

        # 64x128x128 -> 128x64x64
        self.block_2 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=3, 
                                     stride=2, padding=1, multi_channel=True, return_mask=True)
        self.norm_2 = self.normalization_layer(num_features=128)

        # 128x64x64 -> 256x32x32
        self.block_3 = PartialConv2d(in_channels=128, out_channels=256, kernel_size=3, 
                                     stride=2, padding=1, multi_channel=True, return_mask=True)
        self.norm_3 = self.normalization_layer(num_features=256)

        # 256x32x32 -> 512x16x16
        self.block_4 = PartialConv2d(in_channels=256, out_channels=512, kernel_size=3, 
                                     stride=2, padding=1, multi_channel=True, return_mask=True)
        self.norm_4 = self.normalization_layer(num_features=512)

        # 512x16x16 -> 512x8x8
        self.block_5 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, 
                                     stride=2, padding=1, multi_channel=True, return_mask=True)
        self.norm_5 = self.normalization_layer(num_features=512)

        # 512x8x8 -> 512x4x4
        self.block_6 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, 
                                     stride=2, padding=1, multi_channel=True, return_mask=True)
        self.norm_6 = self.normalization_layer(num_features=512)

        # 512x4x4 -> 512x2x2
        self.block_7 = PartialConv2d(in_channels=512, out_channels=512, kernel_size=3, 
                                     stride=2, padding=1, multi_channel=True, return_mask=True)
        self.norm_7 = self.normalization_layer(num_features=512)

        # 1024x4x4 -> 512x4x4
        self.block_8 = PartialConv2d(in_channels=1024, out_channels=512, kernel_size=3, 
                                     stride=1, padding=1, multi_channel=True, return_mask=True)
        self.norm_8 = self.normalization_layer(num_features=512)

        # 1024x8x8 -> 512x8x8
        self.block_9 = PartialConv2d(in_channels=1024, out_channels=512, kernel_size=3, 
                                     stride=1, padding=1, multi_channel=True, return_mask=True)
        self.norm_9 = self.normalization_layer(num_features=512)

        # 1024x16x16 -> 512x16x16
        self.block_10 = PartialConv2d(in_channels=1024, out_channels=512, kernel_size=3, 
                                      stride=1, padding=1, multi_channel=True, return_mask=True)
        self.norm_10 = self.normalization_layer(num_features=512)

        # 768x32x32 -> 256x32x32
        self.block_11 = PartialConv2d(in_channels=768, out_channels=256, kernel_size=3, 
                                      stride=1, padding=1, multi_channel=True, return_mask=True)
        self.norm_11 = self.normalization_layer(num_features=256)

        # 384x64x64 ->128x64x64
        self.block_12 = PartialConv2d(in_channels=384, out_channels=128, kernel_size=3, 
                                      stride=1, padding=1, multi_channel=True, return_mask=True)
        self.norm_12 = self.normalization_layer(num_features=128)

        # 192x128x128 -> 64x128x128
        self.block_13 = PartialConv2d(in_channels=192, out_channels=64, kernel_size=3, 
                                      stride=1, padding=1, multi_channel=True, return_mask=True)
        self.norm_13 = self.normalization_layer(num_features=64)

        # 67x256x256 -> 3x256x256
        self.block_14 = PartialConv2d(in_channels=67, out_channels=3, kernel_size=3, 
                                      stride=1, padding=1, multi_channel=True, return_mask=True)

        if self.freeze_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and name in ['norm_2', 'norm_3', 'norm_4', 
                                                                   'norm_5', 'norm_6', 'norm_7']:
                    module.eval()
            
    def forward(self, x, mask=None):
        # 3x256x256 -> 64x128x128
        x_1, m_1 = self.block_1(x, mask)
        x_1 = self.activation_function_1(x_1)

        # 64x128x128 -> 128x64x64
        x_2, m_2 = self.block_2(x_1, m_1)
        x_2 = self.activation_function_1(self.norm_2(x_2))

        # 128x64x64 -> 256x32x32
        x_3, m_3 = self.block_3(x_2, m_2)
        x_3 = self.activation_function_1(self.norm_3(x_3))

        # 256x32x32 -> 512x16x16
        x_4, m_4 = self.block_4(x_3, m_3)
        x_4 = self.activation_function_1(self.norm_4(x_4))
        
        # 512x16x16 -> 512x8x8
        x_5, m_5 = self.block_5(x_4, m_4)
        x_5 = self.activation_function_1(self.norm_5(x_5))

        # 512x8x8 -> 512x4x4
        x_6, m_6 = self.block_6(x_5, m_5)
        x_6 = self.activation_function_1(self.norm_6(x_6))

        # 512x4x4 -> 512x2x2
        x_7, m_7 = self.block_7(x_6, m_6)
        x_7 = self.activation_function_1(self.norm_7(x_7))

        # 512x2x2 -> 512x4x4 -> 1024x4x4 -> 512x4x4
        out = self.upsample(x_7)
        out_mask = self.upsample(m_7)
        out = torch.cat((out, x_6), dim=1)
        out_mask = torch.cat((out_mask, m_6), dim=1)
        out, out_mask = self.block_8(out, out_mask)
        out = self.activation_function_2(self.norm_8(out))

        # 512x4x4 -> 512x8x8 -> 1024x8x8 -> 512x8x8
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((out, x_5), dim=1)
        out_mask = torch.cat((out_mask, m_5), dim=1)
        out, out_mask = self.block_9(out, out_mask)
        out = self.activation_function_2(self.norm_9(out))

        # 512x8x8 -> 512x16x16 -> 1024x16x16 -> 512x16x16
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((out, x_4), dim=1)
        out_mask = torch.cat((out_mask, m_4), dim=1)
        out, out_mask = self.block_10(out, out_mask)
        out = self.activation_function_2(self.norm_10(out))

        # 512x16x16 -> 512x32x32 -> 768x32x32 -> 256x32x32
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((out, x_3), dim=1)
        out_mask = torch.cat((out_mask, m_3), dim=1)
        out, out_mask = self.block_11(out, out_mask)
        out = self.activation_function_2(self.norm_11(out))

        # 256x32x32 -> 256x64x64 -> 384x64x64 -> 128x64x64
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((out, x_2), dim=1)
        out_mask = torch.cat((out_mask, m_2), dim=1)
        out, out_mask = self.block_12(out, out_mask)
        out = self.activation_function_2(self.norm_12(out))

        # 128x64x64 -> 128x128x128 -> 192x128x128 -> 64x128x128
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((out, x_1), dim=1)
        out_mask = torch.cat((out_mask, m_1), dim=1)
        out, out_mask = self.block_13(out, out_mask)
        out = self.activation_function_2(self.norm_13(out))

        # 64x128x128 -> 64x256x256 -> 67x256x256 -> 3x256x256
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((out, x), dim=1)
        out_mask = torch.cat((out_mask, mask), dim=1)
        out, _ = self.block_14(out, out_mask)

        return out

    @torch.inference_mode()
    def inference(self, 
                  user_img: ImageTorch | ImagePIL,
                  mask: MaskTorch, 
                  crop: int,
                  device: str
                  ) -> MaskTorch:
        self.eval()
        self = self.to(device)

        if mask.shape[0] == 1:
            mask = torch.cat([mask]*3, dim=0)

        masked_img = user_img*mask

        masked_img, dims = functional.prepare_img(img=masked_img, 
                                                  crop=crop)
        mask, _ = functional.prepare_img(img=mask,
                                         crop=crop)

        masked_img = functional.normalize(img=masked_img, 
                                          mean=self.mean,
                                          std=self.std)
        res = []
        n_patches = len(masked_img)
        for i in range(0, n_patches, self.patches_to_device):
            if i+self.patches_to_device > n_patches:
                tmp = masked_img[i:].to(device)
                tmp_mask = mask[i:].to(device)
                
                for patch in self(tmp, tmp_mask):
                    res.append(patch.cpu())
                break

            tmp = masked_img[i:i+self.patches_to_device].to(device)
            tmp_mask = mask[i:i+self.patches_to_device].to(device)
            for patch in self(tmp, tmp_mask):
                res.append(patch.cpu())
        self = self.cpu()
        res = torch.stack(res)
        res = functional.cat_patches(res, dims)
        mask = functional.cat_patches(mask, dims)
        res = functional.unnormalize(res, self.mean, self.std)
        res = user_img*mask + (1-mask)*res
        return res
