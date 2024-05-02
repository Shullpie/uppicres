import torch
from typing import Literal

from utils.types import Network

def get_inference_model(
        inference_options: dict, 
        task: Literal['clr'] | Literal['seg'],
        device: str = 'cpu'
    ) -> Network:

    options = inference_options['nns']
    crop = inference_options['crop']
    model = None

    if task == 'seg':
        from model.modules.archs.unet_wide import UnetWide as SM
        model = SM(
            img_size=crop,
            in_channels=options[f'UnetWide{crop}']['in_channels'],
            out_channels=options[f'UnetWide{crop}']['out_channels'],
            activation_function=options[f'UnetWide{crop}']['activation_function']
        )

        model = model.to(device)
        model_params = torch.load(
            inference_options['nns'][f'UnetWide{crop}']['path'],
            map_location=device
        )

        model.load_state_dict(model_params)
        return model
    else:
        pass
        # TODO logger