import torch
from typing import Literal

from utils.types import Network

def get_inference_model(inference_options: dict, 
                        task: Literal['clr'] | Literal['seg'], 
                        device: str = 'cpu'
                        ) -> Network:
    options = inference_options['nns']
    crop = inference_options['crop']
    model = None

    if task == 'seg':
        from model.modules.archs.unet_256 import Unet256 as SM
        model = SM(
            img_size=crop,
            in_channels=options[f'unet256{crop}']['in_channels'],
            out_channels=options[f'unet256{crop}']['out_channels'],
            activation_function=options[f'unet256{crop}']['activation_function']
        )

        model = model.to(device)
        model_params = torch.load(
            inference_options['nns'][f'unet256{crop}']['path'],
            map_location=device
        )

        model.load_state_dict(model_params)
        return model
    else:
        pass
        # TODO logger