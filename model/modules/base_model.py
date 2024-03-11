from typing import Literal, Optional, Callable, TypeAlias

import torch
from torch.nn import Sigmoid
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import metrics
from .utils import optimizers
from .utils import schedulers
from .models import get_nn, Network
from data.dataloaders import dataloaders

Loss: TypeAlias = float
Metrics: TypeAlias = dict[str, float]

class Trainer():
    
    def __init__(self, 
                 task: Optional[Literal['seg'] | Literal['clr']] = None,
                 device: Optional[Literal['cuda'] | Literal['cpu']] = None,
                 train_loader: Optional[DataLoader] = None, 
                 test_loader: Optional[DataLoader] = None, 
                 transform_data_every_n_epoch: Optional[int] = None,
                 model: Optional[Network] = None,
                 optimizer: Optional[Optimizer] = None, 
                 scheduler=None,
                 n_epoch: int = None,
                 metrics_dict: Optional[dict[str, Callable[[torch.Tensor, torch.Tensor], float]]] = None,
                 criterion: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None) -> None:     
        self.task = task 
        self.device = device
        self.transform_data_every_n_epoch = transform_data_every_n_epoch
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epoch = n_epoch
        self.metrics_dict = metrics_dict
        self.criterion = criterion
        self.is_cropped = None
    
    def init_from_config(self, options: dict) -> None:
        self.task = options['task']
        self.device = options['device']
        self.n_epoch = options['epoch']
        self.model = get_nn(options)
        self.train_loader, self.test_loader = dataloaders.create_dataloaders(options=options)
        self.optimizer = optimizers.get_optimizer(self.model.parameters(), 
                                                  options['nns'][self.task]['models'][self.model.name]['optimizer'])
        self.scheduler = schedulers.get_scheduler(self.optimizer, 
                                                  options['nns'][self.task]['models'][self.model.name]['scheduler'])  # TODO CHECK total_iters and change the value
        self.criterion = metrics.get_criterion(options['nns'][self.task]['criterion'])
        self.transform_data_every_n_epoch = options['transform_data_every_n_epoch']
        self.metrics_dict = metrics.get_metrics(options['nns'][self.task]['metrics'])

    def calc_metrics(self, prediction: torch.Tensor, target: torch.Tensor) -> Metrics:
        calculated_metrics = {}
        prediction = Sigmoid()(prediction)
        target = target.to(dtype=torch.int8)
        for key, metric in self.metrics_dict.items():
            calculated_metrics[key] = metric(prediction, target).item()
        return calculated_metrics
    
    def training_step(self, batch: torch.Tensor) -> tuple[Loss, Metrics]:
        metrics_dict = {}
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        if self.metrics_list:
            metrics_dict = self.calc_metrics(prediction=outputs, target=targets)
        
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()  #TODO?

        return (loss.item(), metrics_dict)

    def train_epoch(self, epoch) -> tuple[Loss, Metrics]:
        self.model.train()

        if self.transform_data_every_n_epoch > 0:
            if epoch%self.transform_data_every_n_epoch == 0:
                self.train_loader.dataset._change_transforms_list()

        loss_by_epoch = 0
        metrics_by_epoch = {}

        for batch in self.train_loader:
            temp_loss_by_epoch, batch_metrics_dict = self.training_step(batch)
            loss_by_epoch += temp_loss_by_epoch

            if self.metrics_dict:
                metrics_by_epoch = self._accumulate_metrics(metrics_by_epoch, batch_metrics_dict)

        dataset_lenght = len(self.train_loader.dataset)
        loss_by_epoch = loss_by_epoch/dataset_lenght
        metrics_by_epoch = self._get_mean_metrics_dict(metrics_by_epoch, dataset_lenght)
        return (loss_by_epoch, metrics_by_epoch)

    def fit(self):
        self._check_attrs()
        self.model.to(self.device)
        
        for epoch in tqdm(range(1, self.n_epoch+1), desc='Epoch: '):
            train_loss, train_metrics = self.train_epoch(epoch)
            #ЗАпись в файл
            test_loss, test_metrics = self.test_epoch()
            #Запись в файл

    def _check_attrs(self) -> None:
        optional = set(['metrics_dict', 'transform_data_every_n_epoch', 'is_cropped'])
        none_list = [key for key, value in vars(self).items() if value is None and key not in optional]
        if none_list:
            raise TypeError(f'Missing {len(none_list)} required argument(s): ' + ', '.join(none_list) + '.')
        self.is_cropped = bool(self.train_loader.dataset.crop)

    @staticmethod
    def _accumulate_metrics(total_metrics_dict: dict[str, float], 
                            batch_metric_dict: dict[str, float]) -> dict[str, float]:
        c_total_metrics_dict = total_metrics_dict.copy()
        if not c_total_metrics_dict:
            return batch_metric_dict.copy()
        
        for key, value in batch_metric_dict.items():
            c_total_metrics_dict[key] += value
        return c_total_metrics_dict

    @staticmethod    
    def _get_mean_metrics_dict(metrics: Metrics, dataset_length: int) -> Metrics:
        c_metrics = metrics.copy()
        for key in metrics.keys():
            c_metrics[key] /= dataset_length
        return c_metrics
