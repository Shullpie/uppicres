import torch
import tqdm

import model.modules.models.base_model as bm


class SegModel(bm.BaseModel):
    
    def __init__(self, options: dict) -> None:     
        super().__init__(options=options)
    
    def training_step(self, batch: torch.Tensor) -> tuple[bm.Loss, bm.Metrics]:
        metrics_dict = {}
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        if self.metrics_list:
            metrics_dict = self._calc_metrics(prediction=outputs, target=targets)
        
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()  #TODO?

        return (loss.item(), metrics_dict)

    def train_epoch(self, epoch) -> tuple[bm.Loss, bm.Metrics]:
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
            self.train_losses += [train_loss]
            test_loss, test_metrics = self.test_epoch()
            self.test_losses += [test_loss]

            if self.telegram_message:
                self._save_chart()
                



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
    def _get_mean_metrics_dict(metrics: bm.Metrics, dataset_length: int) -> bm.Metrics:
        c_metrics = metrics.copy()
        for key in metrics.keys():
            c_metrics[key] /= dataset_length
        return c_metrics
