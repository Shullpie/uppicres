import os
from typing import Optional, TypeAlias, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset

from model.data.processing import augments


ImgTorch: TypeAlias = torch.Tensor
MaskTorch: TypeAlias = torch.Tensor

ImgPIL: TypeAlias = Image.Image
MaskPIL: TypeAlias = Image.Image


class BaseDataSet(Dataset):

    def __init__(self,
                 dataset_type_options: dict,
                 crop: Optional[Literal[256] | Literal[512]]) -> None:
        self.dataset_type_options = dataset_type_options
        self.imgs_path_list, self.masks_path_list = self._get_image_paths()
        self.transforms_list = None

        self.load_to_ram = dataset_type_options['load_to_ram']
        self.normalize = dataset_type_options.get('normalize', None)
        self.images, self.masks = [], []
        if self.load_to_ram:
            self.images, self.masks = self._load_to_ram()

        self.crop = None if crop <= 0 else crop
        if 'transforms' in dataset_type_options: # TODO try with logger
            self.transforms_list = augments._get_transforms_list(dataset_type_options["transforms"])

    def __getitem__(self, idx: int) -> tuple[ImgTorch, MaskTorch]:
        NotImplementedError('Do not use BaseDataSet. Please, use concrete pipeline instand.')

    def __len__(self) -> int:
        return len(self.imgs_path_list)

    def _get_image_paths(self) -> tuple[list[str], list[str]]:
        i_list = os.listdir(self.dataset_type_options['imgs_path'])
        m_list = os.listdir(self.dataset_type_options['masks_path'])
        return sorted(i_list), sorted(m_list)

    def _change_transforms_list(self) -> None:
        try:
            self.transforms_list = augments._get_transforms_list(self.dataset_type_options['transforms'])
        except:
            #TODO logger
            pass

    def _load_to_ram(self) -> tuple[list[ImgPIL], list[MaskPIL]]:
        images, masks = [], []
        for img_filename, mask_filename in zip(self.imgs_path_list, self.masks_path_list):
            images += [Image.open(self.dataset_type_options['imgs_path'] + img_filename).convert('RGB')]
            masks += [Image.open(self.dataset_type_options['masks_path'] + mask_filename).convert('L')]
        return images, masks
