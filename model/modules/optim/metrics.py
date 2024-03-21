from typing import Callable

import torch
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torchmetrics import Dice
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


def get_metrics(metrics_list: list[str], device: str) -> dict[str, Callable]:
    metrics_list = tuple(map(lambda x: x.lower(), metrics_list))
    metrics_dict = {}
    if 'bacc' in metrics_list:
        metrics_dict['BinaryAccuracy'] = BinaryAccuracy().to(device)
    if 'dice' in metrics_list:
        metrics_dict['dice'] = Dice().to(device)
    if 'broc-auc' in metrics_list:
        metrics_dict['roc-auc'] = BinaryAUROC().to(device)

    return metrics_dict


def get_criterion(criterion_options: dict, device: str) -> Callable:
    if len(criterion_options) != 1:
        raise ValueError('"criterion" must contain one element.')
    
    criterion = None
    if 'bce' in criterion_options:
        pos_weight = criterion_options['bce']['pos_weight']
        criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weight).to(device))

    else:
        raise NotImplementedError(f'criterion={criterion_options.keys()[0]} is not recognized. Check your config file.')
    
    return criterion


def calc_metrics(prediction: torch.Tensor, 
                 target: torch.Tensor,
                 metrics: dict[str, Callable]) -> dict[str, float]:
    output = {}
    prediction = Sigmoid()(prediction)
    target = target.to(dtype=torch.int8)
    for label, metric in metrics.items():
        output[label] = metric(prediction, target)

    if output == {}:
        raise NotImplementedError("metrics was not found. Check your config file.")
    
    return output
