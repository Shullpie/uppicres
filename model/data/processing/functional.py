import torch
from torchvision.transforms import Resize, CenterCrop
from torchvision.transforms.functional import crop

Img = torch.Tensor


def crop_into_nxn(img: Img, n: int) -> torch.Tensor:
    _, img_h, img_w = img.shape
    if (img_h % n != 0) or (img_w % n != 0):
        raise ValueError(f'The image side is not divisible by n={n}. h={img_h}, w={img_w}.')
    
    patches: list[torch.Tensor] = []
    for colomn in range(0, img_h, n):
        for line in range(0, img_w, n):
            patches.append(crop(img, top=colomn, left=line, height=n, width=n))
    return torch.stack(patches)


def resize_multiples_n(img: Img, n: int) -> Img:
    _, img_h, img_w = img.shape
    if (img_w % n == 0) and (img_h % n == 0): 
        return img

    if img_w % n != 0:
        if (n - img_w%n) <= 64:
            img_w = img_w + (n - img_w%n)
            img = Resize(size=(img_h, img_w), antialias=None)(img)
        else:
            img_w = img_w-img_w%n
            img = CenterCrop(size=(img_h, img_w))(img)
        
    if img_h % n != 0:
        if (n - img_h%n) <= 64:
            img_h = img_h + (n - img_h%n)
            img = Resize(size=(img_h, img_w), antialias=None)(img)
        else:
            img_h = img_h - img_h%n
            img = CenterCrop(size=(img_h, img_w))(img)
    return img
 
