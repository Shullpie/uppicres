import logging
import logging.config

from model import model
from utils import options


OPTIONS, LOGGER_OPTIONS = options.get_options()
logging.config.dictConfig(LOGGER_OPTIONS)


def main():
    mode = OPTIONS.get('mode', None)
    
    if mode == 'train':
        model.train(OPTIONS)
    
    elif mode == 'inference':
        pass

    else:
        raise NotImplementedError(f'Mode "{mode}" is not recognized. Check your config file.')


if __name__ == "__main__":
    main()
