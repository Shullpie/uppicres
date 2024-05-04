import os
import logging

import torch
import matplotlib
import matplotlib.pyplot as plt

from model.modules.optim import metrics
from model.modules.optim import optimizers
from model.modules.optim import schedulers
from model.modules.models import get_train_model
from model.data.dataloaders import dataloaders
from utils.types import Metrics
matplotlib.use('Agg')


class BaseModel():

    def __init__(self, options: dict) -> None: 
        # main
        self.task = options.get('task')
        self.device = options.get('device')
        self.n_epoch = options.get('epoch')
        self.crop = options.get('crop', None)
        self.is_cropped = bool(self.crop)
        self.model = get_train_model(options)
        
        # data
        self.train_loader, self.test_loader = dataloaders.create_dataloaders(options=options)
        self.transform_data_every_n_epoch = options.get('transform_data_every_n_epoch', 0)

        # optimization and metrics
        self.nns_options = options['nns']['models'][self.model.name]
        self.optimizer = optimizers.get_optimizer(self.model.parameters(), self.nns_options.get('optimizer'))
        self.scheduler = schedulers.get_scheduler(self.optimizer, self.nns_options.get('scheduler'))
        self.criterion = metrics.get_criterion(options['nns']['criterion'], self.device)
        self.metrics_dict = metrics.get_metrics(options['nns']['metrics'], self.device)

        # logging
        self.checkpoints_path = options.get('checkpoints_path')
        self.logger = logging.getLogger(__name__)    
        self.make_checkpoint_every_n_epoch = options.get('make_checkpoint_every_n_epoch', 0)
        self.telegram_message = options.get('telegram_send_logs', False)

        self.curr_epoch = 1
        self.train_losses = []
        self.test_losses = []
        
    def fit(self):
        NotImplementedError('Do not use BaseModel. Please, use concrete pipeline instand.')

    def load_model_from_checkpoint(self, path: str) -> None:
        self.model.to(self.device)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.curr_epoch = checkpoint['epoch'] + 1

        del checkpoint
        
    def _reset_metrics(self) -> None:
        for metric in self.metrics_dict.values():
            metric.reset()
    
    def _save_model_state(self, epoch: int, chart: bool = False) -> None:
        path = self.checkpoints_path + \
            f'/{self.model.name}_{type(self.optimizer).__name__}_{type(self.scheduler).__name__}' + \
            f'_crop{self.crop}/'
        path = path.lower()
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        if chart:
            self._save_chart(path + 'chart.png')
        try:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'test_losses': self.test_losses
                },
                path+f'{epoch}_epoch.pt'
            )
        except Exception as ex:
            self.logger.error(ex, exc_info=True)

        self.model.to(self.device)

    def _save_chart(self, path):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(range(len(self.train_losses)), self.train_losses,
                 range(len(self.test_losses)), self.test_losses)
        plt.title('Loss function -> min')
        plt.legend(['train', 'test'])

        try:
            plt.savefig(path)
        except Exception as ex:
            self.logger.error(ex, exc_info=True)
        
        plt.close(fig)

    def _send_telegram_message(self, 
                               epoch: int,
                               train_metrics_dict: Metrics,
                               test_metrics_dict: Metrics
                               ) -> None:
        self._save_chart('logs/temp/chart.png')
        train_metrics_str = '\n'.join(map(lambda x: f'{x[0]}: {x[1]:.3f}', train_metrics_dict.items()))
        test_metrics_str = '\n'.join(map(lambda x: f'{x[0]}: {x[1]:.3f}', test_metrics_dict.items()))

        try:
            self.logger.info(
                f'*📊Epoch {epoch}/{self.n_epoch}*\n'
                f'*🔴Train metrics:*\n_Loss: {self.train_losses[-1]:.3f}\n{train_metrics_str}_\n\n'
                f'*🟢Test metrics:*\n_Loss: {self.test_losses[-1]:.3f}\n{test_metrics_str}_\n\n'
                '*⚙️Training params: *' 
                f'*LR: * {self.scheduler._last_lr}'
                '%'
                r'D:\workspace\projects\uppicres\logs\temp\chart.png',
                extra={'telegram': True}
            )
        except Exception as ex:
            self.logger.error(ex, exc_info=True)

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
            f'Model\'s name: {self.model.name}\n'
            f'Tranform data every n epoches: {self.transform_data_every_n_epoch}\n'
            f'Make checpoints every n epoches: {self.make_checkpoint_every_n_epoch}\n'
            f'Send Telegram logs: {self.telegram_message}\n\n'
            f'Criterion: {self.criterion}\n'
            f'Metrics: {self.metrics_dict.values()}\n'
            f'> Optimizer: {self.optimizer}\n\n'
            f'> Scheduler: {self.scheduler}\n'
            f'Scheduler params: {self.scheduler.state_dict()}\n',
            extra={'telegram': False}
        ) 

    @torch.inference_mode()  
    def _calc_metrics(self, prediction: torch.Tensor, target: torch.Tensor) -> Metrics:
        calculated_metrics = {}
        target = target.to(dtype=torch.int8)
        for key, metric in self.metrics_dict.items():
            calculated_metrics[key] = metric(prediction, target).item()
        return calculated_metrics

    @staticmethod
    @torch.inference_mode()
    def _accumulate_metrics(total_metrics_dict: dict[str, float], 
                            batch_metric_dict: dict[str, float]
                            ) -> dict[str, float]:
        if not total_metrics_dict:
            return batch_metric_dict.copy()
        
        for key, value in batch_metric_dict.items():
            total_metrics_dict[key] += value
        return total_metrics_dict

    @staticmethod   
    @torch.inference_mode() 
    def _get_mean_metrics_dict(metrics: Metrics, n_batches: int) -> Metrics:
        for key in metrics.keys():
            metrics[key] /= n_batches
        return metrics
    
