from typing import Callable, NamedTuple
from PIL.Image import Image

from torch import Tensor
import torchvision.transforms as T
from numpy.random import choice, uniform

Transformation = Callable
Img = Image
Mask = Image


class PTransformation(NamedTuple):
    p: float
    transformation: Transformation[[Image], Image]


def _get_transforms_list(
        transforms_options: dict[str, dict]) -> list[Transformation | PTransformation]:
    
    transforms_list: list[Transformation | PTransformation] = []
    for key, options in transforms_options.items():
        key = key.lower()

        if key == 'grayscale':
            transforms_list.append(
                PTransformation(
                    options['p'],
                    T.Grayscale(
                        num_output_channels=options['num_output_channels']
                    )
                )
            )

        elif key == 'randomequalize':
            transforms_list.append(T.RandomEqualize(**options))

        elif key == 'randomhorizontalflip':
            transforms_list.append(
                PTransformation(
                    options['p'],
                    T.RandomHorizontalFlip(p=1)
                )
            )

        elif key == 'randomverticalflip':
            transforms_list.append(
                PTransformation(
                    options['p'],
                    T.RandomVerticalFlip(p=1)
                )
            )

        elif key == 'randomadjustsharpness':
            transforms_list.append(
                T.RandomAdjustSharpness(
                    p=options['p'],
                    sharpness_factor=choice(options['sharpness_factor'])
                )
            )

        elif key == 'randomposterize':
            transforms_list.append(
                T.RandomPosterize(
                    p=options['p'],
                    bits=choice(options['bits'])
                )
            )

        elif key == 'colorjitter':
            transforms_list.append(
                PTransformation(
                    options['p'],
                    T.ColorJitter(
                        brightness=choice(options['brightness']),
                        contrast=choice(options['contrast']),
                        saturation=choice(options['saturation']),
                        hue=choice(options['hue']),
                    )
                )
            )

        else:
            raise ValueError(f'Transformation "{key}" is not valid.')
    return transforms_list


def _transform(img: Img, mask: Mask, 
               transformation: Transformation | PTransformation) -> tuple[Img, Mask]:
    if type(transformation) is PTransformation:
        if uniform(low=0, high=1) < transformation.p:
            img = transformation.transformation(img)
            if isinstance(transformation.transformation, T.RandomHorizontalFlip) or \
               isinstance(transformation.transformation, T.RandomVerticalFlip):
                mask = transformation.transformation(mask)
    else:
        img = transformation(img)
    return img, mask
      

def apply_transforms(img: Tensor, mask: Tensor, 
                     transforms_list: list[Transformation | PTransformation]
                     ) -> tuple[Img, Mask]:
    for transformation in transforms_list:
        img, mask = _transform(img, mask, transformation)
    return T.ToTensor()(img), T.ToTensor()(mask)
