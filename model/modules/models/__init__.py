from typing import TypeAlias
import torch.nn as nn


Network: TypeAlias = nn.Module


def get_nn(options: dict) -> Network:
    nn_model = None
    task = options.get('task', None)
    model_str = options.get('nn_model', None)
    crop = options.get('crop')
    
    if task == 'seg':
        if model_str == 'SegUnet':
            from model.modules.archs.seg_unet import SegUnet as NN
        else:
            raise NotImplementedError(f'NN "{model_str}" is not recognized. Check your config file.')
        
    elif task == 'clr':
        pass
    else:
        raise NotImplementedError(f'Task "{task}" is not supported. Check your config file.')
    
    nn_model = NN(img_size=crop)
    nn_model.init_from_config(options['nns'][task]['models'][model_str])
    return nn_model