from typing import Callable
import torch
from torch.nn import BCEWithLogitsLoss
from torchmetrics import Dice
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import MeanSquaredError as MSE

from model.modules.optim.inpainting_loss import InpaintingLoss


def get_metrics(metrics_list: list[str], device: str) -> dict[str, Callable]:
    metrics_list = tuple(map(lambda x: x.lower(), metrics_list))
    metrics_dict = {}
    if 'bacc' in metrics_list:
        metrics_dict['BinaryAccuracy'] = BinaryAccuracy().to(device)
    if 'dice' in metrics_list:
        metrics_dict['dice'] = Dice().to(device)
    if 'broc-auc' in metrics_list:
        metrics_dict['roc-auc'] = BinaryAUROC().to(device)
    if 'ssim' in metrics_list:
        metrics_dict['ssim'] = SSIM().to(device)
    if 'mse' in metrics_list:
        metrics_dict['mse'] = MSE().to(device)

    return metrics_dict


def get_criterion(criterion_options: dict, device: str) -> Callable:
    if len(criterion_options) != 1:
        raise ValueError('"criterion" must contain one element.')
    
    criterion = None
    if 'bce' in criterion_options:
        pos_weight = criterion_options['bce']['pos_weight']
        criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weight).to(device))
    elif 'inpainting' in criterion_options:
        lambdas = criterion_options['inpainting']['lambdas']
        criterion = InpaintingLoss(lambdas).to(device)
    else:
        raise NotImplementedError(f'criterion={criterion_options.keys()[0]} is not recognized. Check your config file.')
    
    return criterion
