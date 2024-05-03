from typing import TypeAlias
import torch
import torch.nn as nn


Network: TypeAlias = nn.Module


def get_train_model(options: dict) -> Network:
    nn_model = None
    task = options.get('task', None)
    model_str = options['nn_model'].lower()
    crop = options.get('crop')

    if task == 'seg':
        if model_str == 'unetwide':
            from model.modules.archs.unet_wide import UnetWide as NN
            nn_model = NN(img_size=crop)
            nn_model.init_from_config(options['nns']['models'][model_str])

        elif model_str == 'segnet':
            from model.modules.archs.segnet import SegNet as NN
            nn_model = NN()
        else:
            raise NotImplementedError(f'NN "{model_str}" is not recognized. Check your config file.')

    elif task == 'clr':
        pass
        
    else:
        raise NotImplementedError(f'Task "{task}" is not supported. Check your config file.')
    return nn_model