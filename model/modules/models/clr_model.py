import torch
from tqdm import tqdm

import model.modules.models.base_model as bm
from utils.types import Loss, Metrics


class ClrModel(bm.BaseModel):
    def __init__(self, options: dict) -> None:
        super().__init__(options=options)

    def training_step(self, batch: list[torch.Tensor, torch.Tensor]) -> tuple[Loss, Metrics]:
        inputs, masks, targets = batch
        metrics_dict = {}

        self.optimizer.zero_grad()
        inputs = inputs.to(self.device)
        masks = masks.to(self.device)
        outputs = self.model(inputs, masks)

        targets = targets.to(self.device)
        loss = self.criterion(inputs, masks, outputs, targets)
        
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            if self.metrics_dict:
                metrics_dict = self._calc_metrics(prediction=outputs, target=targets)
        return (loss.item(), metrics_dict)

    def train_epoch(self, epoch: int) -> tuple[Loss, Metrics]:
        self.model.train()

        loss_by_epoch = 0
        metrics_by_epoch = {}

        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.n_epoch} - Train', ncols=200):
            batch_loss, batch_metrics_dict = self.training_step(batch)
            loss_by_epoch += batch_loss

            if self.metrics_dict:
                metrics_by_epoch = self._accumulate_metrics(metrics_by_epoch, batch_metrics_dict)
            
        n_batches = len(self.train_loader)
        loss_by_epoch = loss_by_epoch/n_batches
        metrics_by_epoch = self._get_mean_metrics_dict(metrics_by_epoch, n_batches)
        return (loss_by_epoch, metrics_by_epoch)

    @torch.inference_mode()
    def testing_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[Loss, Metrics]:
        inputs, masks, targets = batch
        metrics_dict = {}

        inputs = inputs.to(self.device)
        masks = masks.to(self.device)
        outputs = self.model(inputs, masks)

        targets = targets.to(self.device)
        loss = self.criterion(inputs, masks, outputs, targets)

        if self.metrics_dict:
            metrics_dict = self._calc_metrics(prediction=outputs, target=targets)
        return (loss.item(), metrics_dict)

    def test_epoch(self, epoch: int) -> tuple[Loss, Metrics]:
        self.model.eval()

        loss_by_epoch = 0
        metrics_by_epoch = {}

        for batch in tqdm(self.test_loader, desc=f'Epoch {epoch}/{self.n_epoch} - Test ', ncols=200):
            temp_loss_by_epoch, batch_metrics_dict = self.testing_step(batch)
            loss_by_epoch += temp_loss_by_epoch

            if self.metrics_dict:
                metrics_by_epoch = self._accumulate_metrics(metrics_by_epoch, batch_metrics_dict)

        n_batches = len(self.test_loader)
        loss_by_epoch = loss_by_epoch/n_batches
        metrics_by_epoch = self._get_mean_metrics_dict(metrics_by_epoch, n_batches)
        return (loss_by_epoch, metrics_by_epoch)

    def fit(self) -> None:
        self.model.to(self.device)
        self._get_trainer_info()

        for epoch in range(self.curr_epoch, self.n_epoch+1):
            train_loss, train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            test_loss, test_metrics = self.test_epoch(epoch)
            self.test_losses.append(test_loss)

            self.scheduler.step(train_loss)
            self._reset_metrics()
            self.train_loader.dataset._shuffle_masks()

            if self._send_telegram_message:
                self._send_telegram_message(epoch=epoch,
                                            train_metrics_dict=train_metrics,
                                            test_metrics_dict=test_metrics)
                
            if (self.make_checkpoint_every_n_epoch 
                and epoch % self.make_checkpoint_every_n_epoch == 0):
                self._save_model_state(epoch=epoch, chart=True)

            if (self.transform_data_every_n_epoch 
                and epoch % self.transform_data_every_n_epoch == 0):
                self.train_loader.dataset._change_transforms_list()
        
        self._save_model_state(self.n_epoch, chart=True)
        self._send_telegram_message(epoch=epoch,
                                    train_metrics_dict=train_metrics,
                                    test_metrics_dict=test_metrics)
