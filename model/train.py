def get_train_model(options: dict, arc: str = None):
    task = options.get('task', None)
    if task == 'seg':
        from model.modules.models.seg_model import SegModel as M
    elif task == 'clr':
        pass
    else:
        # TODO LOGGER raise NotImplementedError(f'Task "{task}" is not recognized. Check your config file.')
        pass
    return M(options=options)
            
            
def train(options):
    model = get_train_model(options)
    if options['checkpoint']:
        model.load_model_from_checkpoint(options['checkpoint'])
    model.fit()

        




    