
import torch
import torchvision.transforms as T
from typing import Iterable, Literal, Optional
from torchvision.transforms.functional import crop

from utils.types import (
    ImageTorch, MaskTorch,
    CroppedImageTorch, CroppedMaskTorch,
    ImagePIL
)


def crop_into_nxn(img: ImageTorch | MaskTorch, 
                  n: int
                  ) -> tuple[CroppedImageTorch | CroppedMaskTorch, tuple[int, int]]:
    _, img_h, img_w = img.shape
    if n is None or (img_h == img_w == n):
        return img, (1, 1)

    if (img_h % n != 0) or (img_w % n != 0):
        raise ValueError(f'The image side is not divisible by n={n}. h={img_h}, w={img_w}.')

    patches = []
    for colomn in range(0, img_h, n):
        for line in range(0, img_w, n):
            patches.append(crop(img, top=colomn, left=line, height=n, width=n))
    return torch.stack(patches), (img_h//256, img_w//256)


def resize_multiples_n(img: ImageTorch | ImagePIL, 
                       n: int
                       ) -> ImageTorch | ImagePIL:
    if isinstance(img, ImagePIL):
        img = T.ToTensor()(img)
    _, img_h, img_w = img.shape
    if n is None or (img_w % n == 0) and (img_h % n == 0):
        return img

    if img_w % n != 0:
        if (n - img_w%n) <= img_w*0.1:
            img_w = img_w + (n - img_w%n)
            img = T.Resize(size=(img_h, img_w), antialias=None)(img)
        else:
            img_w = img_w-img_w%n
            img = T.CenterCrop(size=(img_h, img_w))(img)

    if img_h % n != 0:
        if (n - img_h%n) <= img_h*0.1:
            img_h = img_h + (n - img_h%n)
            img = T.Resize(size=(img_h, img_w), antialias=None)(img)
        else:
            img_h = img_h - img_h%n
            img = T.CenterCrop(size=(img_h, img_w))(img)
    return img


def prepare_img(img: ImageTorch, 
                mean: Optional[int | Iterable] = None, 
                std: Optional[int | Iterable] = None,
                crop: Literal[256] | Literal[512] = 256
                ) -> ImageTorch:  # TODO Lireral, optional mean,std
    
    if isinstance(img, ImagePIL):
        img = T.ToTensor()(img)
    if mean is not None and std is not None:
        img = T.Normalize(mean, std)(img)
    img = resize_multiples_n(img, crop)
    img, dims = crop_into_nxn(img, crop)
    return img, dims


def cat_patches(patches: CroppedMaskTorch, 
                dims: tuple[int, int]
                ) -> MaskTorch:
    res = []
    x, y = dims
    for i in range(0, y*x, y):
        res.append(torch.cat(tuple(patches[i:i+y]), dim=-1))
    return torch.cat(tuple(res), dim=-2)


def unnormalize(img, mean, std):
    invTrans = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1/s for s in std]),
                          T.Normalize(mean=[-m for m in mean], std=[1., 1., 1.])])

    return invTrans(img)
