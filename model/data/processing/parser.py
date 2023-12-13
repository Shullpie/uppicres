import inspect
from data.processing import augments 
from numpy.random import choice 

def _get_transforms_list(options):
    transforms_list = []
    for transformation, params in options.items():
        if transformation in augments.VALID_TRANSFORMATIONS:
            if params["p"]:
                t_params = params.copy()
                item = {
                    "p": t_params.pop("p"),
                    "mask": augments.VALID_TRANSFORMATIONS[transformation][1]
                }
                transformation = augments.VALID_TRANSFORMATIONS[transformation][0]
                
                if "p" in inspect.getfullargspec(transformation).args:
                    t_params["p"] = 1

                for param, value in t_params.items():
                    if isinstance(value, list):
                        t_params[param] = choice(value)
                        
                item["transformation"] = transformation(**t_params)
                transforms_list.append(item)
        else:
            raise NotImplementedError(f"""Transformation "{transformation}" is not recognized.""")
    return transforms_list

def apply_transforms(img, mask, transforms_list):
    for transformation in transforms_list:
        img, mask = augments.transform(img, mask, transformation)
    return img, mask