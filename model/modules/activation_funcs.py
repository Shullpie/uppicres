def get_activation_function(func_name):
    if func_name == "relu":
        from torch.nn import ReLU
        return ReLU(inplace=True)
    elif func_name == "gelu":
        from torch.nn import GELU
        return GELU()
    else:
        raise NotImplementedError(f'Activation function "{func_name}" is not recognized.')