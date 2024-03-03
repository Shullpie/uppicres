import torch
from typing import Callable
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torchmetrics import Dice
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


def get_metrics(metrics_list: list[str]) -> dict[str, float]:
    if not metrics_list:
        raise ValueError('"metrics_list" is empty. Check your config file.')
    
    metrics_list = tuple(map(lambda x: x.lower(), metrics_list))
    metrics_dict = {}
    if 'bacc' in metrics_list:
        metrics_dict['BinaryAccuracy'] = BinaryAccuracy()
    if 'dice' in metrics_list:
        metrics_dict['dice'] = Dice()
    if 'broc-auc' in metrics_list:
        metrics_dict['roc-auc'] = BinaryAUROC()

    return metrics_dict


def get_loss_func(loss_fn: dict) -> Callable:
    if len(loss_fn) != 1:
        raise ValueError('"loss_fn" must contain one element.')
    
    criterion = None
    if 'bce' in loss_fn:
        pos_weight = loss_fn['bce']['pos_weight']
        criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weight))

    if criterion is None:
        raise NotImplementedError(f'loss_fn = {loss_fn} is not recognized. Check your config file.')
    
    return criterion


def calc_metrics(prediction: torch.Tensor, 
                 target: torch.Tensor,
                 metrics: dict[str, Callable]) -> dict[str, float]:
    output = {}
    prediction = Sigmoid()(prediction)
    target = target.to(dtype=torch.uint8)
    for label, metric in metrics.items():
        output[label] = metric(prediction, target)
    return output
