import os
from typing import Optional, TypeAlias

import torch
from PIL import Image
from torch.utils.data import Dataset

from model.data.processing import seg_augments
from model.data.processing import functional


Img: TypeAlias = torch.Tensor
Mask: TypeAlias = torch.Tensor

ImgP: TypeAlias = Image.Image
MaskP: TypeAlias = Image.Image


class SegDataSet(Dataset):

    def __init__(self, dataset_type_options: dict, crop: Optional[int]) -> None:
        self.dataset_type_options = dataset_type_options
        self.imgs_path_list, self.masks_path_list = self._get_image_paths()
        self.transforms_list = None

        self.load_to_ram = dataset_type_options['load_to_ram']
        self.normalize = dataset_type_options.get('normalize', None)
        self.images, self.masks = [], []
        if self.load_to_ram:
            self.images, self.masks = self._load_images(self.imgs_path_list, self.masks_path_list)
            
        self.crop = None if crop <= 0 else crop
        if 'transforms' in dataset_type_options: 
            self.transforms_list = seg_augments._get_transforms_list(dataset_type_options["transforms"])  

    def __getitem__(self, idx: int) -> tuple[Img, Mask]:
        img, mask = None, None
        if self.load_to_ram:
            img = self.images[idx]
            mask = self.masks[idx]
        else:
            img = Image.open(self.dataset_type_options['imgs_path'] + self.imgs_path_list[idx]).convert('RGB')
            mask = Image.open(self.dataset_type_options['masks_path'] + self.masks_path_list[idx]).convert('L')
        
        img, mask = seg_augments.apply_transforms(img=img, 
                                                  mask=mask, 
                                                  transforms_list=self.transforms_list,
                                                  normalize=self.normalize)
        if self.crop is not None:
            img, _ = functional.crop_into_nxn(img=img, n=self.crop)
            mask, _ = functional.crop_into_nxn(img=mask, n=self.crop)
        return img, mask

    def __len__(self) -> int:
        return len(self.imgs_path_list)

    def _get_image_paths(self) -> tuple[list[str], list[str]]:
        i_list = os.listdir(self.dataset_type_options['imgs_path']) 
        m_list = os.listdir(self.dataset_type_options['masks_path'])
        return sorted(i_list), sorted(m_list)
    
    def _change_transforms_list(self) -> None:
        self.transforms_list = seg_augments._get_transforms_list(self.dataset_type_options['transforms'])

    def _load_images(self, 
                     imgs_path_list: list[str], 
                     masks_path_list: list[str]) -> tuple[list[ImgP], list[MaskP]]:
        images, masks = [], []
        for img_filename, mask_filename in zip(imgs_path_list, masks_path_list):
            images += [Image.open(self.dataset_type_options['imgs_path'] + img_filename).convert('RGB')]
            masks += [Image.open(self.dataset_type_options['masks_path'] + mask_filename).convert('L')]
        return images, masks
        
 
