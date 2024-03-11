import argparse
import yaml
import torch
from modules.base_model import Trainer
from data.dataloaders import dataloaders
from data.datasets.seg_dataset import SegDataSet
from data.visualization import show_image
from PIL import Image
import torchvision.transforms as T
from data.processing.functional import crop_into_nxn
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-options', type=str, required=True, help='Path to options file.')
    args = parser.parse_args()
    option_path = args.options

    with open(option_path, 'r') as file_option:
        options = yaml.safe_load(file_option)

    d = SegDataSet(options['datasets']['seg_dataset']['train'], crop=0)
    img = d[0][0]
    print(len(d.images))
    print(isinstance(img, Image.Image))
    show_image(d[0][0])

    
        # a = Trainer()
    # a.init_from_config(options)
    # print(a.train_loader.dataset.transforms_list)
    # train_loader, _ = dataloaders.create_dataloaders(options)
    # a.init_from_config(options=options)
    # a._check_attrs()
    # # print(a.is_cropped)
    # # print(a.metrics_dict)
    
    # i = iter(train_loader)
    # batch = next(i)
    # inputs, target = batch
    # print(a.model(inputs).shape)
        
    


if __name__=="__main__":
    main()
    