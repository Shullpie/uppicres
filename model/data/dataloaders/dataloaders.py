from torch.utils.data import DataLoader
from typing import NamedTuple

from model.data.datasets import create_datasets


class Dataloaders(NamedTuple):
    train_loader: DataLoader
    test_loader: DataLoader


def create_dataloaders(options: dict) -> Dataloaders:
    task = options['task']
    train, test = create_datasets(options)
    train_loader = DataLoader(train, **options['datasets'][f'{task}_dataset']['dataloader']['train'])
    test_loader = DataLoader(test, **options['datasets'][f'{task}_dataset']['dataloader']['test'])
    return Dataloaders(train_loader=train_loader, test_loader=test_loader)
