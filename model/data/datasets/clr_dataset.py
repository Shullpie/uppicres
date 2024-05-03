import torch
import torchvision.transforms as T
from PIL import Image
from typing import Optional, Literal

from model.data.processing import functional, augments
from model.data.datasets.base_dataset import BaseDataSet, ImgTorch


class ClrDataSet(BaseDataSet):

    def __init__(self,
                 dataset_type_options: dict,
                 crop: Optional[Literal[256] | Literal[512]]
                 ) -> None:
        super().__init__(dataset_type_options, crop)

    def __getitem__(self, idx: int) -> tuple[ImgTorch, ImgTorch]:
        clear_img, mask = None, None
        if self.load_to_ram:
            clear_img = self.images[idx]
            mask = self.masks[idx]
        else:
            clear_img = Image.open(self.dataset_type_options['imgs_path'] + self.imgs_path_list[idx]).convert('RGB')
            mask = Image.open(self.dataset_type_options['masks_path'] + self.masks_path_list[idx]).convert('L')

        clear_img, mask = augments.apply_transforms(img=clear_img,
                                                    mask=mask,
                                                    transforms_list=self.transforms_list,
                                                    normalize=self.normalize)
        mask = T.RandomInvert(p=1)(mask)
        img = torch.cat((clear_img, mask), dim=0)
        if self.crop is not None:
            clear_img, _ = functional.crop_into_nxn(img=clear_img, n=self.crop)
            img, _ = functional.crop_into_nxn(img=img, n=self.crop)

        return img, clear_img
