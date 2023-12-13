from torch.utils.data import Dataset
from data.processing import parser
import torchvision.transforms as T
from PIL import Image
import os


class SegDataSet(Dataset):
    def __init__(self, 
                 options: dict
                 ):
        self.options = options
        self.imgs_list, self.masks_list = self._get_image_paths()
        self.transforms_list = None
        if "transforms" in options: 
            self.transforms_list = parser._get_transforms_list(options["transforms"])

    def __getitem__(self, idx):
        img = Image.open(self.options["imgs_path"] + self.imgs_list[idx]).convert("RGB")
        mask = Image.open(self.options["masks_path"] + self.masks_list[idx]).convert("L")
        if self.transforms_list:
            img, mask = parser.apply_transforms(img, mask, self.transforms_list)

        return T.ToTensor()(img), T.ToTensor()(mask)

    def __len__(self):
        return len(self.imgs_list)    

    def _get_image_paths(self):
        i_list, m_list = os.listdir(self.options["imgs_path"]), os.listdir(self.options["masks_path"])
        return  sorted(i_list), sorted(m_list)
