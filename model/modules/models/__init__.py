from utils.types import Network


def get_train_model(options: dict) -> Network:
    task = options.get('task', None)
    model_str = options['nn_model'].lower()
    crop = options.get('crop')

    nn_model = None
    nn_options = options['nns']['models'][model_str]
    if task == 'seg':
        if model_str == 'unetwide':
            from model.modules.archs.unet_wide import UnetWide as NN
            nn_model = NN(img_size=crop)
            nn_model.init_from_config(options['nns']['models'][model_str])

        elif model_str == 'unet256':
            from model.modules.archs.unet_256 import Unet256 as NN
            nn_model = NN(nn_options)

        elif model_str == 'segnet':
            from model.modules.archs.segnet import SegNet as NN
            nn_model = NN()
            
        elif model_str == 'fcn_resnet50':
            from model.modules.archs.fcn_resnet50 import FCNResNet as NN
            nn_model = NN(nn_options)
        else:
            raise NotImplementedError(f'NN "{model_str}" is not recognized. Check your config file.')

    elif task == 'clr':
        if model_str == 'pnet_256':
            from model.modules.archs.pnet_256 import PConvNet256 as NN
            nn_model = NN(nn_options)
        else:
            raise NotImplementedError(f'NN "{model_str}" is not recognized. Check your config file.')
    else:
        raise NotImplementedError(f'Task "{task}" is not supported. Check your config file.')
    return nn_model