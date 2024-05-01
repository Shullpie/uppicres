from typing import Optional, Literal
from PIL import Image

from model.data.processing import functional, augments
from model.data.datasets.base_dataset import (BaseDataSet,
                                              ImgTorch, MaskTorch)

class SegDataSet(BaseDataSet):

    def __init__(self,
                 dataset_type_options: dict,
                 crop: Optional[Literal[256] | Literal[512]]) -> None:
        super().__init__(dataset_type_options, crop)

    def __getitem__(self, idx: int) -> tuple[ImgTorch, MaskTorch]:
        img, mask = None, None
        if self.load_to_ram:
            img = self.images[idx]
            mask = self.masks[idx]
        else:
            img = Image.open(self.dataset_type_options['imgs_path'] + self.imgs_path_list[idx]).convert('RGB')
            mask = Image.open(self.dataset_type_options['masks_path'] + self.masks_path_list[idx]).convert('L')

        img, mask = augments.apply_transforms(img=img,
                                              mask=mask,
                                              transforms_list=self.transforms_list,
                                              normalize=self.normalize)
        if self.crop is not None:
            img, _ = functional.crop_into_nxn(img=img, n=self.crop)
            mask, _ = functional.crop_into_nxn(img=mask, n=self.crop)
        return img, mask
