import os
from typing import TypeAlias

import torch
from torch.nn import Sigmoid
import matplotlib.pyplot as plt

from model.modules.optim import metrics
from model.modules.optim import optimizers
from model.modules.optim import schedulers
from model.modules.models import get_nn
from model.data.dataloaders import dataloaders
from utils.options import get_logger

Loss: TypeAlias = float
Metrics: TypeAlias = dict[str, float]


class BaseModel:

    def __init__(self, options: dict) -> None: 
        self.task = options.get('task')
        self.device = options.get('device')
        self.n_epoch = options.get('epoch')
        self.crop = options.get('crop', None)
        self.is_cropped = bool(self.crop)
        self.logger = get_logger(self.task)    
        self.train_loader, self.test_loader = dataloaders.create_dataloaders(options=options)
        self.model = get_nn(options)
        self.nns_options = options['nns'][self.task]['models'][self.model.name]
        self.optimizer = optimizers.get_optimizer(self.model.parameters(), self.nns_options.get('optimizer'))
        self.scheduler = schedulers.get_scheduler(self.optimizer, self.nns_options.get('scheduler'))
        self.transform_data_every_n_epoch = options.get('transform_data_every_n_epoch', 0)
        self.criterion = metrics.get_criterion(options['nns'][self.task]['criterion'])
        self.metrics_dict = metrics.get_metrics(options['nns'][self.task]['metrics'])
        self.telegram_message = options.get('telegram_send_logs', False)

        self.logs_path = options.get('logs_path')
        self._check_attrs()
        self._get_trainer_info()

        self.train_losses = []
        self.test_losses = []
        
    def fit(self):
        NotImplementedError('Do not use BaseModel. Please, use concrete pipeline instand.')
    
    def _calc_metrics(self, prediction: torch.Tensor, target: torch.Tensor) -> Metrics:
        calculated_metrics = {}
        prediction = Sigmoid()(prediction)
        target = target.to(dtype=torch.int8)
        for key, metric in self.metrics_dict.items():
            calculated_metrics[key] = metric(prediction, target).item()
        return calculated_metrics
    
    def _check_attrs(self) -> None:
        optional = set(['metrics_dict', 'transform_data_every_n_epoch'])
        none_list = [key for key, value in vars(self).items() if value is None and key not in optional]
        if none_list:
            raise TypeError(f'Missing {len(none_list)} required argument(s): ' + ', '.join(none_list) + '.')
    
    def _save_loss_and_metrics(self) -> None:  #TODO Do
        pass
    
    def _save_model_state(self, epoch: int) -> None:
        path = self.logs_path + \
            f'states/{self.model.name}_{type(self.optimizer).__name__}_{type(self.scheduler).__name__}/'
        
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'opimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            },
            path+f'{epoch}_epoch.pt'
        )

    def load_model_from_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.model.load_state_dict(path)

    def _save_chart(self):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(range(len(self.train_losses)), self.train_losses,
                 range(len(self.test_losses)), self.test_losses)
        plt.title('Loss function -> min')
        plt.legend(['train', 'test'])
        plt.savefig(self.logs_path + 'temp/chart.png')

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
    def _get_batch_metrics_dict(metrics: Metrics, dataset_length: int) -> Metrics:
        c_metrics = metrics.copy()
        for key in metrics.keys():
            c_metrics[key] /= dataset_length
        return c_metrics
    
    def _send_telegram_message(self, 
                               epoch: int,
                               train_metrics_dict: Metrics,
                               test_metrics_dict: Metrics):
        train_metrics_str = '\n'.join(map(lambda x: f'{x[0]}: {x[1]}', train_metrics_dict.items()))
        test_metrics_str = '\n'.join(map(lambda x: f'{x[0]}: {x[1]}', test_metrics_dict.items()))
        self.logger.info(
            f'*📊Epoch {epoch}/{self.n_epoch}*\n'
            f'*🔴Train metrics:*\n_Loss: {self.train_losses[-1]}\n{train_metrics_str}_\n\n'
            f'*🟢Test metrics:*\n_Loss: {self.test_losses[-1]}\n{test_metrics_str}_\n\n'
            '*⚙️Training params: *'  #TODO Scheduler lr,  
            '%'
            r'D:\workspace\projects\uppicres\logs\temp\chart.png'

        )

    def _get_trainer_info(self) -> None:
        self.logger.info(
            '---=|Main info|=---\n'
            'Model info: \n'
            f'Task: {self.task}\n'
            f'Device: {self.device}\n'
            f'Number of epoches: {self.n_epoch}\n'
            f'Image size: {self.crop}x{self.crop}\n\n\n'
            '---=|DataLoader\'s info|=---\n'
            '> Train Dataloader\n'
            f'Batch size: {self.train_loader.batch_size}\n'
            f'Number of batches: {len(self.train_loader)}\n\n'
            '> Test Dataloader\n'
            f'Batch size: {self.test_loader.batch_size}\n'
            f'Number of batches: {len(self.test_loader)}\n\n\n'
            '---=|DataSet\'s info|=---\n'
            f'Loaded to RAM: {self.train_loader.dataset.load_to_ram}\n\n'
            '> Train DataSet\n'
            f'Number of images: {len(self.train_loader.dataset)}\n\n'
            '> Test DataSet\n'
            f'Number of images: {len(self.test_loader.dataset)}\n\n\n'
            '---=|NN\'s info|=---\n'
            f'Model\'s name: {self.model.name}\n\n'
            f'Criterion: {self.criterion}\n'
            f'Metrics: {self.metrics_dict.values()}\n'
            f'> Optimizer: {self.optimizer}\n\n'
            f'> Scheduler: {self.scheduler}\n'
            f'Scheduler params: {self.scheduler.state_dict()}\n'
        ) 
