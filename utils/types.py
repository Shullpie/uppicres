import torch
import typing

from PIL import Image

ImagePIL = Image.Image
MaskPIL = Image.Image

ImageTorch = torch.Tensor
MaskTorch = torch.Tensor
CroppedImageTorch = torch.Tensor
CroppedMaskTorch = torch.Tensor

Network = torch.nn.Module
