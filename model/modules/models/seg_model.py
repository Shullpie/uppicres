import torch
from tqdm import tqdm

import model.modules.models.base_model as bm


class SegModel(bm.BaseModel):
    
    def __init__(self, options: dict) -> None:     
        super().__init__(options=options)
    
    def training_step(self, batch: torch.Tensor) -> tuple[bm.Loss, bm.Metrics]:
        metrics_dict = {}
        inputs, targets = batch
        
        if self.is_cropped:
            n = inputs.shape[0]*inputs.shape[1]
            c_i, c_t = inputs.shape[2], targets.shape[2]
            img_size = self.crop

            inputs = inputs.reshape((n, c_i, img_size, img_size))
            targets = targets.reshape((n, c_t, img_size, img_size))

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if self.metrics_dict:
            metrics_dict = self._calc_metrics(prediction=outputs, target=targets)
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss)

        return (loss.item(), metrics_dict)

    def train_epoch(self, epoch) -> tuple[bm.Loss, bm.Metrics]:
        self.model.train()

        if self.transform_data_every_n_epoch > 0:
            if epoch%self.transform_data_every_n_epoch == 0:
                self.train_loader.dataset._change_transforms_list()

        loss_by_epoch = 0
        metrics_by_epoch = {}

        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.n_epoch} - Train'):
            batch_loss, batch_metrics_dict = self.training_step(batch)
            loss_by_epoch += batch_loss
            if self.metrics_dict:
                metrics_by_epoch = self._accumulate_metrics(metrics_by_epoch, batch_metrics_dict)

        dataset_lenght = len(self.train_loader.dataset)
        loss_by_epoch = loss_by_epoch/dataset_lenght
        metrics_by_epoch = self._get_mean_metrics_dict(metrics_by_epoch, dataset_lenght)
        return (loss_by_epoch, metrics_by_epoch)

    def testing_step(self, batch) -> tuple[bm.Loss, bm.Metrics]:
        metrics_dict = {}
        inputs, targets = batch

        if self.is_cropped:
            n = inputs.shape[0]*inputs.shape[1]
            c_i, c_t = inputs.shape[2], targets.shape[2]
            img_size = self.crop

            inputs = inputs.reshape((n, c_i, img_size, img_size))
            targets = targets.reshape((n, c_t, img_size, img_size))

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.logger.info(f'requires_grad {loss.requires_grad}')

        if self.metrics_dict:
            metrics_dict = self._calc_metrics(prediction=outputs, target=targets)

        return (loss.item(), metrics_dict)

    def test_epoch(self, epoch) -> tuple[bm.Loss, bm.Metrics]:
        self.model.eval()

        loss_by_epoch = 0
        metrics_by_epoch = {}

        for batch in tqdm(self.test_loader, desc=f'Epoch {epoch}/{self.n_epoch} - Test'):
            temp_loss_by_epoch, batch_metrics_dict = self.testing_step(batch)
            loss_by_epoch += temp_loss_by_epoch

            if self.metrics_dict:
                metrics_by_epoch = self._accumulate_metrics(metrics_by_epoch, batch_metrics_dict)

        dataset_lenght = len(self.test_loader.dataset)
        loss_by_epoch = loss_by_epoch/dataset_lenght
        metrics_by_epoch = self._get_mean_metrics_dict(metrics_by_epoch, dataset_lenght)
        return (loss_by_epoch, metrics_by_epoch)
    
    def fit(self):
        self._check_attrs()
        self.model.to(self.device)
        
        for epoch in range(1, self.n_epoch+1):
            train_loss, train_metrics = self.train_epoch(epoch)
            self.train_losses += [train_loss]
            test_loss, test_metrics = self.test_epoch(epoch)
            self.test_losses += [test_loss]

            if self.metrics_dict and self.telegram_message:
                self._save_chart()
                self._send_telegram_message(epoch=epoch,
                                            train_metrics_dict=train_metrics,
                                            test_metrics_dict=test_metrics)
                
            if (self.make_checkpoint_every_n_epoch 
                and epoch % self.make_checkpoint_every_n_epoch == 0):
                self._save_model_state(epoch=epoch)

        self._save_model_state(self.n_epoch)
