import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, invert, to_pil_image

from model.modules import archs
from bot.utils import reserved
from model.data.processing import functional


@reserved
@torch.inference_mode()
def infernce(user_img: Image.Image, inference_options: dict, todo: dict[str, bool]):
    user_img = to_tensor(user_img)
    device = inference_options.get('device', 'cpu')
    crop = inference_options.get('crop', 256)
    user_img = functional.resize_multiples_n(user_img, crop)

    if todo['clr']:
        model = archs.get_inference_model(
            inference_options=inference_options,
            task='seg'
        )

        mask = model.inference(
            user_img=user_img, 
            crop=crop,
            device=device
        )
        del model
        mask = invert(mask)
        model = archs.get_inference_model(
            inference_options=inference_options,
            task='clr'
        )
        res = model.inference(user_img=user_img, 
                              mask=mask,
                              crop=crop,
                              device=device)
        del model
        return to_pil_image(res, 'RGB')
