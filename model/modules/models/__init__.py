from typing import TypeAlias
import torch
import torch.nn as nn


Network: TypeAlias = nn.Module


def get_train_model(options: dict) -> Network:
    nn_model = None
    task = options.get('task', None)
    model_str = options.get('nn_model', None)
    if isinstance(model_str, str):
        model_str = model_str.lower()
    crop = options.get('crop')

    if task == 'seg':
        if model_str == 'unet':
            from model.modules.archs.unet import Unet as NN
        elif model_str == 'unetwide':
            from model.modules.archs.unet_wide import UnetWide as NN
        else:
            raise NotImplementedError(f'NN "{model_str}" is not recognized. Check your config file.')

    elif task == 'clr':
        if model_str == 'unet':
            from model.modules.archs.unet import Unet as NN
        elif model_str == 'unetwide':
            from model.modules.archs.unet_wide import UnetWide as NN
        else:
            raise NotImplementedError(f'NN "{model_str}" is not recognized. Check your config file.')
        
    else:
        raise NotImplementedError(f'Task "{task}" is not supported. Check your config file.')

    nn_model = NN(img_size=crop)
    nn_model.init_from_config(options['nns'][task]['models'][model_str])
    return nn_model