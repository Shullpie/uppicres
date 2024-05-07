import torch
import torch.nn as nn
from torchvision import models
from torch_pconv import PConv2d

from model.modules.optim import activation_funcs


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PConvNet256(nn.Module):
    def __init__(self, nn_options: dict):
        super(PConvNet256, self).__init__()
        self.activation_function_1 = activation_funcs.get_activation_function(nn_options['activation_function_1'])
        self.activation_function_2 = activation_funcs.get_activation_function(nn_options['activation_function_2'])
        self.normalization_layer = nn.BatchNorm2d
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)
        self.activatop
        # 3x256x256 -> 64x128x128
        self.block_1 = PConv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2)

        # 64x128x128 -> 128x64x64
        self.block_2 = PConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.norm_2 = self.normalization_layer(num_features=128)

        # 128x64x64 -> 256x32x32
        self.block_3 = PConv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.norm_3 = self.normalization_layer(num_features=256)

        # 256x32x32 -> 512x16x16
        self.block_4 = PConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.norm_4 = self.normalization_layer(num_features=512)

        # 512x16x16 -> 512x8x8
        self.block_5 = PConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.norm_5 = self.normalization_layer(num_features=512)

        # 512x8x8 -> 512x4x4
        self.block_6 = PConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.norm_6 = self.normalization_layer(num_features=512)

        # 512x4x4 -> 512x2x2
        self.block_7 = PConv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.norm_7 = self.normalization_layer(num_features=512)

        # 1024x4x4 -> 512x4x4
        self.block_8 = PConv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.norm_8 = self.normalization_layer(num_features=512)

        # 1024x8x8 -> 512x8x8
        self.block_9 = PConv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.norm_9 = self.normalization_layer(num_features=512)

        # 1024x16x16 -> 512x16x16
        self.block_10 = PConv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.norm_10 = self.normalization_layer(num_features=512)

        # 768x32x32 -> 256x32x32
        self.block_11 = PConv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm_11 = self.normalization_layer(num_features=256)

        # 384x64x64 ->128x64x64
        self.block_12 = PConv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm_12 = self.normalization_layer(num_features=128)

        # 192x128x128 -> 64x128x128
        self.block_13 = PConv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm_13 = self.normalization_layer(num_features=64)

        # 67x256x256 -> 3x256x256
        self.block_14 = PConv2d(in_channels=67, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask=None):
        # 3x256x256 -> 64x128x128
        x_1, m_1 = self.block_1(x, mask)
        x_1 = self.activation_function_1(self.norm_1(x_1))

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
        x_5 = self.activation_function_1(self.norm_4(x_5))

        # 512x8x8 -> 512x4x4
        x_6, m_6 = self.block_6(x_5, m_5)
        x_6 = self.activation_function_1(self.norm_4(x_6))

        # 512x4x4 -> 512x2x2
        x_7, m_7 = self.block_7(x_6, m_6)
        x_7 = self.activation_function_1(self.norm_4(x_7))

        # 512x2x2 -> 512x4x4 -> 1024x4x4 -> 512x4x4
        out = self.upsample(x_7)
        out_mask = self.upsample(m_7)
        out = torch.cat((x_6, out), dim=1)
        out_mask = torch.cat((m_6, out_mask), dim=1)

        out, out_mask = self.block_8(out, out_mask)
        out = self.activation_function_2(self.norm_8(out), negative_slope=0.2)

        # 512x4x4 -> 512x8x8 -> 1024x8x8 -> 512x8x8
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((x_5, out), dim=1)
        out_mask = torch.cat((m_5, out_mask), dim=1)
        out, out_mask = self.block_9(out, out_mask)
        out = self.activation_function_2(self.norm_9(out), negative_slope=0.2)

        # 512x8x8 -> 512x16x16 -> 1024x16x16 -> 512x16x16
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((x_4, out), dim=1)
        out_mask = torch.cat((m_4, out_mask), dim=1)
        out, out_mask = self.block_10(out, out_mask)
        out = self.activation_function_2(self.norm_10(out), negative_slope=0.2)

        # 512x16x16 -> 512x32x32 -> 768x32x32 -> 256x32x32
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((x_3, out), dim=1)
        out_mask = torch.cat((m_3, out_mask), dim=1)
        out, out_mask = self.block_11(out, out_mask)
        out = self.activation_function_2(self.norm_11(out), negative_slope=0.2)

        # 256x32x32 -> 256x64x64 -> 384x64x64 -> 128x64x64
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((x_2, out), dim=1)
        out_mask = torch.cat((m_2, out_mask), dim=1)
        # del x_3, m_3
        out, out_mask = self.block_12(out, out_mask)
        out = self.activation_function_2(self.norm_12(out), negative_slope=0.2)

        # 128x64x64 -> 128x128x128 -> 192x128x128 -> 64x128x128
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((x_1, out), dim=1)
        out_mask = torch.cat((m_1, out_mask), dim=1)
        # del x_3, m_3
        out, out_mask = self.block_13(out, out_mask)
        out = self.activation_function_2(self.norm_13(out), negative_slope=0.2)

        # 64x128x128 -> 64x256x256 -> 67x256x256 -> 3x256x256
        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((x, out), dim=1)
        out_mask = torch.cat((mask, out_mask), dim=1)
        out, _ = self.block_14(out, out_mask)
        # out = torch.tanh(out) # TODO tanh

        return out
