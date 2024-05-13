import torchvision.transforms as T
import numpy as np
from PIL import Image

from model.data.processing import augments
from model.data.datasets.base_dataset import BaseDataSet
from utils.types import ImageTorch, ImagePIL, MaskPIL


class ClrDataSet(BaseDataSet):

    def __init__(self,
                 options: dict,
                 dataset_type: dict,
                 ) -> None:
        super().__init__(options, dataset_type)

    def __getitem__(self, idx: int) -> tuple[ImageTorch, ImageTorch]:
        gt_image, mask = None, None
        if self.load_to_ram:
            gt_image = self.images[idx]
            mask = self.masks[idx]
        else:
            gt_image = Image.open(self.dataset_type['imgs_path'] + self.imgs_path_list[idx]).convert('RGB')
            mask = Image.open(self.dataset_type['masks_path'] + self.masks_path_list[idx]).convert('RGB')

        gt_image, mask = augments.apply_transforms(img=gt_image,
                                                   mask=mask,
                                                   transforms_list=self.transforms_list,
                                                   normalize=self.normalize)
        mask = T.RandomRotation((0, 45))(mask)
        mask = T.RandomInvert(p=1)(mask)
        return gt_image*mask, mask, gt_image
    
    def _load_to_ram(self) -> tuple[list[ImagePIL], list[MaskPIL]]:
        images, masks = [], []
        for img_filename, mask_filename in zip(self.imgs_path_list, self.masks_path_list):
            images += [Image.open(self.dataset_type['imgs_path'] + img_filename).convert('RGB')]
            masks += [Image.open(self.dataset_type['masks_path'] + mask_filename).convert('RGB')]
        return images, masks
    
    def _shuffle_masks(self) -> None:
        if self.load_to_ram:
            np.random.shuffle(self.masks)
        else:
            np.random.shuffle(self.masks_path_list)
