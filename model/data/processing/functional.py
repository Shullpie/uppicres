import torch
import torchvision.transforms as T
from torchvision.transforms.functional import crop
from PIL import Image


def crop_into_nxn(img: torch.Tensor, n: int) -> torch.Tensor:
    _, img_h, img_w = img.shape
    if img_h == img_w == 256:
        return img
    
    if (img_h % n != 0) or (img_w % n != 0):
        raise ValueError(f'The image side is not divisible by n={n}. h={img_h}, w={img_w}.')
    
    patches: list[torch.Tensor] = []
    for colomn in range(0, img_h, n):
        for line in range(0, img_w, n):
            patches.append(crop(img, top=colomn, left=line, height=n, width=n))
    return torch.stack(patches), (img_w//256, img_h//256)


def resize_multiples_n(img: torch.Tensor, n: int) -> torch.Tensor:
    _, img_h, img_w = img.shape
    if (img_w % n == 0) and (img_h % n == 0): 
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


def prepare_img(img: Image.Image, mean, std, crop):  # TODO Lireral, optional mean,std
    if mean is not None and std is not None:
        img = T.Normalize(mean, std)(img)

    img = resize_multiples_n(img, crop)
    img, dims = crop_into_nxn(img, crop)
    return img, dims

# def concatenate_patches(patches, dims):
#     img_size = patches[0].shape[-1]
#     res = []
#     for i in range(0, dims[0], dims[1]):
#         res.append(torch.cat(tuple(patches[i:i+img_size//self.crop]), dim=-1))
#     return torch.cat(tuple(res), dim=-2).unsqueeze(0)
