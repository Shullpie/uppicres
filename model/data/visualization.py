import matplotlib.pyplot as plt
from PIL import Image
import torch 
from torchvision.transforms import ToPILImage

def show_image(img: Image):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


def show_batch(batch: torch.Tensor):
    fig = plt.figure(figsize=(8, 4*batch[0].shape[0]))
    for i in range(batch[0].shape[0]):
        plt.subplot(batch[0].shape[0], 2, 2*i+1)
        plt.axis('off')
        plt.imshow(ToPILImage()(batch[0][i]))
        
        plt.subplot(batch[0].shape[0], 2, 2*i+2)
        plt.axis('off')
        plt.imshow(ToPILImage()(batch[1][i]))
    plt.show()
