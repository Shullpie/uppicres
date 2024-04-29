from typing import TypeAlias
import torch
import torch.nn as nn


Network: TypeAlias = nn.Module


def get_train_model(options: dict) -> Network:
    nn_model = None
    task = options.get('task', None)
    model_str = options.get('nn_model', None)
    crop = options.get('crop')
    
    if task == 'seg':
        if model_str == 'SegUnet':
            from model.modules.archs.seg_unet import SegUnet as NN
        elif model_str == 'SegUnetWide':
            from model.modules.archs.seg_unet_wide import SegUnetWide as NN
        else:
            raise NotImplementedError(f'NN "{model_str}" is not recognized. Check your config file.')
        
    elif task == 'clr':
        pass
    else:
        raise NotImplementedError(f'Task "{task}" is not supported. Check your config file.')
    
    nn_model = NN(img_size=crop)
    nn_model.init_from_config(options['nns'][task]['models'][model_str])
    return nn_model


def get_inference_model(inference_options: dict, device: str, label: str) -> Network | tuple[Network]:
    options = inference_options['nns']
    crop = inference_options['crop']

    if label == 'S':
        from model.modules.archs.seg_unet_wide import SegUnetWide as SM
        sm = SM(img_size=crop, 
                in_channels=options[f'SegUnetWide{crop}']['in_channels'], 
                out_channels=options[f'SegUnetWide{crop}']['out_channels'],
                activation_function=options[f'SegUnetWide{crop}']['activation_function'])
        
        sm = sm.to(device)
        model_params = torch.load(inference_options['nns'][f'SegUnetWide{crop}']['path'])
        
        # if device == 'cpu':
        #     model_params = {k: v.cpu() for k, v in model_params['']} # TODO statedict to cpu


        sm.load_state_dict(model_params)
        return sm
    else:
        pass
        # TODO logger