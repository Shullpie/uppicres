import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt


def show_image(img: Image.Image | torch.Tensor) -> None:
    fig = plt.figure(figsize=(12, 12)) 
    if isinstance(img, torch.Tensor):
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
