import os

from PIL import Image
from torch.utils.data import Dataset

from ..processing import seg_augments
from ..processing import functional


class SegDataSet(Dataset):
    def __init__(self, dataset_type_options: dict, crop=None):
        self.dataset_type_options = dataset_type_options
        self.imgs_list, self.masks_list = self._get_image_paths()
        self.transforms_list = None
        self.crop = None if crop <= 0 else crop
        if "transforms" in dataset_type_options: 
            self.transforms_list = seg_augments._get_transforms_list(dataset_type_options["transforms"])

    def __getitem__(self, idx):
        img = Image.open(self.dataset_type_options["imgs_path"] + self.imgs_list[idx]).convert("RGB")
        mask = Image.open(self.dataset_type_options["masks_path"] + self.masks_list[idx]).convert("L")
        if self.transforms_list is not None:
            img, mask = seg_augments.apply_transforms(img=img, 
                                                      mask=mask, 
                                                      transforms_list=self.transforms_list)
        if self.crop is not None:
            img = functional.crop_into_nxn(img=img, n=self.crop)
            mask = functional.crop_into_nxn(img=mask, n=self.crop)
        return img, mask

    def __len__(self):
        return len(self.imgs_list)

    def _get_image_paths(self):
        i_list = os.listdir(self.dataset_type_options["imgs_path"]) 
        m_list = os.listdir(self.dataset_type_options["masks_path"])
        return sorted(i_list), sorted(m_list)
 
