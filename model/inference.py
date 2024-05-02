import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from model.modules import archs


@torch.inference_mode()
def infernce(user_img: Image.Image, inference_options: dict, todo: dict[str, bool]):
    user_img = to_tensor(user_img)
    device = inference_options.get('device', 'cpu')
    
    if todo['clr']:
        model = archs.get_inference_model(
            inference_options=inference_options,
            device=device,
            task='seg'
        )

        mask = model.inference(
            user_img=user_img, 
            inference_options=inference_options,
            device=device
        )
        return mask
