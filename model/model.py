def train(options):
    task = options.get('task', None)

    if task == 'seg':
        from model.modules.models.seg_model import SegModel as M
    
    elif task == 'clr':
        pass

    else:
        raise NotImplementedError(f'Task "{task}" is not recognized. Check your config file.')
    
    model = M(options)
    if options['checkpoint']:
        model.load_model_from_checkpoint(options['checkpoint'])
        model.scheduler.patience = 6
    model.fit()
    
