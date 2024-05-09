def get_train_model(options: dict, arc: str = None):
    task = options.get('task', None)
    if task == 'seg':
        from model.modules.models.seg_model import SegModel as M
        return M(options=options)
    elif task == 'clr':
        from model.modules.models.clr_model import ClrModel as M
        return M(options=options)
    else:
        # TODO LOGGER raise NotImplementedError(f'Task "{task}" is not recognized. Check your config file.')
        pass
            
            
def train(options):
    model = get_train_model(options)
    if options['checkpoint']:
        model.load_model_from_checkpoint(options['checkpoint'])
        # for g in model.optimizer.param_groups:
        #     g['lr'] = 0.0001
        # model.scheduler._last_lr[0] = 1e-4
        # model.scheduler.patience = 12
    model.fit()

        



    
