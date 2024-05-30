import logging
import logging.config

import asyncio

from model import train
from utils import options
from bot import bot

OPTIONS, LOGGER_OPTIONS = options.get_options()
logging.config.dictConfig(LOGGER_OPTIONS)


def main():
    mode = OPTIONS.get('mode', None)
    
    if mode == 'train':
        train.train(OPTIONS)
    
    elif mode == 'inference':
        asyncio.run(bot.start())
        
    else:
        raise NotImplementedError(f'Mode "{mode}" is not recognized. Check your config file.')


if __name__ == "__main__":
    main()
