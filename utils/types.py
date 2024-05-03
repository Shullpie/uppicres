import torch
import typing
from torch.utils.data import DataLoader, Dataset

from PIL import Image

ImagePIL = Image.Image
MaskPIL = Image.Image

ImageTorch = torch.Tensor
MaskTorch = torch.Tensor
ImgOrMask = torch.Tensor

CroppedImageTorch = torch.Tensor
CroppedMaskTorch = torch.Tensor

Network = torch.nn.Module
Transformation = typing.Callable
Loss = float
Metrics = dict[str, float]


class PTransformation(typing.NamedTuple):
    p: float
    transformation: Transformation[[ImgOrMask], ImgOrMask]


class Datasets(typing.NamedTuple):
    train_set: Dataset
    test_set: Dataset


class Dataloaders(typing.NamedTuple):
    train_loader: DataLoader
    test_loader: DataLoader
    
