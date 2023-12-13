import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler, loss_func, metrics_funcs=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.metrics_funcs = metrics_funcs

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "train_loss"
            },
        }
    
    def training_step(self, train_batch):
        x, y = train_batch
        y_pred = self.forward(x)

        loss = self.loss_func(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        #TODO https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        # self.log_dict(self.metrics_funcs, on_step=False, on_epoch=True, prog_bar=True)

    # def validation_step(self, val_batch):
    #     x, y = val_batch
    #     y_pred = self.forward(x)
    #     # self.log_dict(self. metrics_funcs, on_step=False, on_epoch=True)
        
        


        

        

