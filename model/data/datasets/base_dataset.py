import os
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset

from model.data.processing import augments
from utils.types import (ImageTorch, MaskTorch,
                         ImagePIL, MaskPIL)


class BaseDataSet(Dataset):

    def __init__(
        self,
        options: dict,
        mode: Literal['train'] | Literal['test']
    ) -> None:
        self.images, self.masks = [], []
        self.crop = options.get('crop', None)
        self.mode_options = options['datasets'][mode]
        self.load_to_ram = self.mode_options['load_to_ram']
        self.normalize = options.get('datasets', None).get('normalize', None)
        self.imgs_path_list, self.masks_path_list = self._get_image_paths()

        if self.load_to_ram:
            self.images, self.masks = self._load_to_ram()

        self.transforms_list = None
        if 'transforms' in self.mode_options:  # TODO try with logger
            self.transforms_list = augments._get_transforms_list(self.mode_options["transforms"])

    def __getitem__(self, idx: int) -> tuple[ImageTorch, MaskTorch]:
        NotImplementedError('Do not use BaseDataSet. Please, use concrete pipeline instand.')

    def __len__(self) -> int:
        return len(self.imgs_path_list)

    def _get_image_paths(self) -> tuple[list[str], list[str]]:
        i_list = os.listdir(self.mode_options['imgs_path'])
        m_list = os.listdir(self.mode_options['masks_path'])
        return sorted(i_list), sorted(m_list)

    def _change_transforms_list(self) -> None:
        try:
            self.transforms_list = augments._get_transforms_list(self.mode_options['transforms'])
        except Exception as ex:
            # TODO logger
            pass

    def _load_to_ram(self) -> tuple[list[ImagePIL], list[MaskPIL]]:
        images, masks = [], []
        for img_filename, mask_filename in zip(self.imgs_path_list, self.masks_path_list):
            images += [Image.open(self.mode_options['imgs_path'] + img_filename).convert('RGB')]
            masks += [Image.open(self.mode_options['masks_path'] + mask_filename).convert('L')]
        return images, masks
