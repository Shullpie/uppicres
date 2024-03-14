import logging
import logging.config

from utils import options
from model import model
OPTIONS, LOGGER_OPTIONS = options.get_options()
logging.config.dictConfig(LOGGER_OPTIONS)

def main():
    model.test(OPTIONS)

if __name__=="__main__":
    main()
