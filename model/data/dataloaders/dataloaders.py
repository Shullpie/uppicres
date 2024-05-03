from torch.utils.data import DataLoader

from model.data.datasets import create_datasets
from utils.types import Dataloaders


def create_dataloaders(options: dict) -> Dataloaders:
    train, test = create_datasets(options)
    train_loader = DataLoader(train, **options['dataloaders']['train'])
    test_loader = DataLoader(test, **options['dataloaders']['test'])
    return Dataloaders(train_loader=train_loader, test_loader=test_loader)
