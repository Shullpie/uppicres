def get_activation_function(func_name: str):
    if func_name == "relu":
        from torch.nn import ReLU
        return ReLU(inplace=True)
    elif func_name == "gelu":
        from torch.nn import GELU
        return GELU()
    elif func_name == 'leakyrelu':
        from torch.nn import LeakyReLU
        return LeakyReLU(0.2, inplace=True)  #TODO check ns
    else:
        raise NotImplementedError(f'Activation function "{func_name}" is not recognized.')
