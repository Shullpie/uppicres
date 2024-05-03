import torch
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

from utils.types import ImagePIL, ImageTorch


def show_image(img: ImageTorch | ImagePIL) -> None:
    fig = plt.figure(figsize=(12, 12)) 
    if isinstance(img, ImageTorch):
        img = ToPILImage()(img)
    plt.imshow(img)
    plt.show()


def show_batch(batch: torch.Tensor) -> None:
    fig = plt.figure(figsize=(8, 4*batch[0].shape[0]))
    for i in range(batch[0].shape[0]):
        plt.subplot(batch[0].shape[0], 2, 2*i+1)
        plt.axis('off')
        plt.imshow(ToPILImage()(batch[0][i]))
        
        plt.subplot(batch[0].shape[0], 2, 2*i+2)
        plt.axis('off')
        plt.imshow(ToPILImage()(batch[1][i]))
    plt.show()
