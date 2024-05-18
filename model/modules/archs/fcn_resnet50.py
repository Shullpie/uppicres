import torch
import torch.nn as nn
from torchvision.transforms.functional import hflip
from torchvision.models.segmentation import fcn_resnet50

from model.data.processing import functional
from utils.types import ImageTorch, MaskTorch


class FCNResNet(nn.Module):
    name = 'fcn_resnet50'

    def __init__(self, options):
        super().__init__()
        self.out_channels = options.get('out_channels')
        print(self.out_channels)
        self.model = fcn_resnet50(num_classes=self.out_channels)

    def forward(self, x):
        return self.model(x)

    @torch.inference_mode()
    def inference(self, 
                  user_img: ImageTorch, 
                  inference_options: dict, 
                  device: str
                  ) -> MaskTorch:
        self.eval()
        res = []

        options = inference_options['seg']
        crop = inference_options.get('crop', None)
        step = inference_options.get('patches_to_device', 2)
        pos_label_threshold = options.get('pos_label_threshold', 0.5)

        mean = options.get('normalize', None).get('mean', None)
        std = options.get('normalize', None).get('std', None)

        flipped_image = hflip(user_img)

        user_img, dims = functional.prepare_img(
            img=user_img, 
            mean=mean,
            std=std,
            crop=crop
        )  
        flipped_image, _ = functional.prepare_img(
            img=flipped_image, 
            mean=mean,
            std=std,
            crop=crop
        )    
        
        n_patches = len(user_img)
        for img in [user_img, flipped_image]:
            mask = []
            for i in range(0, n_patches, step):
                if i+step > n_patches:
                    tmp = img[i:].to(device)
                    
                    for patch in self(tmp)['out']:
                        mask.append(patch.cpu())
                    break

                tmp = img[i:i+step].to(device)
                for patch in self(tmp)['out']:
                    mask.append(patch.cpu())
            res.append(torch.stack(mask))
        mask1, mask2 = res
        del res

        mask1 = torch.sigmoid(functional.cat_patches(mask1, dims))
        mask2 = torch.sigmoid(functional.cat_patches(mask2, dims))
        mask2 = hflip(mask2)

        mask1 = mask1 > pos_label_threshold
        mask2 = mask2 > pos_label_threshold
        return torch.logical_or(mask1, mask2).to(torch.float32)
