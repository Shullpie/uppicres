from numpy.random import uniform
from torchvision.transforms import v2

VALID_TRANSFORMATIONS = {
    "Grayscale": (v2.Grayscale, 0),
    "ColorJitter": (v2.ColorJitter, 0),
    "RandomInvert": (v2.RandomInvert, 0),
    "RandomEqualize": (v2.RandomEqualize, 0),
    "RandomPosterize": (v2.RandomPosterize, 0),
    "RandomAdjustSharpness": (v2.RandomAdjustSharpness, 0),
    "RandomHorizontalFlip": (v2.RandomHorizontalFlip, 1),
    "RandomVerticalFlip": (v2.RandomVerticalFlip, 1)
}

def transformation(options):
    def transformation_options(func):
        def tansformation_function(img, mask):
            result = (img, mask)
            
            if uniform(low=0, high=1) < options["p"]:
                result = func(img, mask, options)
            return result
        return tansformation_function
    return transformation_options

def transform(img, mask, options):

    @transformation(options) # options
    def get_transform(img, mask, transform):
        transform = options["transformation"]
        img = transform(img)
        if options["mask"]:
            mask = transform(mask)
        return img, mask
    return get_transform(img, mask)