import torch
from torch.optim import Optimizer


def get_optimizer(model_params: torch.Tensor, option_optimizer: dict) -> Optimizer:
    name = option_optimizer.get('name', None)
    if name is None:
        raise NotImplementedError(
            'Optimizer is None. Please, add to config file')
    name = name.lower()

    optimizer = None
    if name == "sgd":
        lr = float(option_optimizer.get("lr"))
        momentum = float(option_optimizer.get("momentum", 0.0))
        weight_decay = float(option_optimizer.get('weight_decay', 0.0))
        nesterov = option_optimizer.get('nesterov', False)
        dampening = float(option_optimizer.get('dampening', 0.0))

        optimizer = torch.optim.SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            dampening=dampening
        )
        
    elif name in ('adam', 'adamw'):
        lr = float(option_optimizer.get('lr'))
        beta1 = float(option_optimizer.get('beta1', 0.9))
        beta2 = float(option_optimizer.get('beta2', 0.999))
        weight_decay = float(option_optimizer.get('weight_decay', 0.0))
        amsgrad = option_optimizer.get('amsgrad', False)

        Class_optimizer = torch.optim.AdamW if name == 'adamw' else torch.optim.Adam

        optimizer = Class_optimizer(
            model_params, 
            lr=lr, 
            betas=(beta1, beta2), 
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

    elif name in 'adamax':
        lr = float(option_optimizer.get('lr'))
        beta1 = float(option_optimizer.get('beta1', 0.9))
        beta2 = float(option_optimizer.get('beta2', 0.999))
        weight_decay = float(option_optimizer.get('weight_decay', 0.0))

        Class_optimizer = torch.optim.Adamax 
        optimizer = Class_optimizer(
            model_params, 
            lr=lr, 
            betas=(beta1, beta2), 
            weight_decay=weight_decay
        )
        
    else:
        raise NotImplementedError(
            f'Optimizer [{name}] is not recognized. optimizers.py doesn\'t know {[name]}')

    return optimizer
