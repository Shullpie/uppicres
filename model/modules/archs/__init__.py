import torch
from typing import Literal

from utils.types import Network

def get_inference_model(inference_options: dict, 
                        task: Literal['clr'] | Literal['seg'], 
                        ) -> Network:
    options = inference_options['nns']
    model = None

    if task == 'seg':
        seg_model = inference_options['seg_model']

        from model.modules.archs.unet_256 import Unet256 as SM
        model = SM(options[seg_model])

        model_params = torch.load(
            inference_options['nns'][seg_model]['path'],
            map_location='cpu'
        )

        model.load_state_dict(model_params)
        return model
    
    elif task == 'clr':
        clr_model = inference_options['clr_model']

        from model.modules.archs.pnet_256 import PConvNet256 as CM
        model = CM(options[clr_model])
        model_params = torch.load(
            inference_options['nns'][clr_model]['path'],
            map_location='cpu'
        )
        
        model.load_state_dict(model_params)
        return model
    else:
        pass
        # TODO logger