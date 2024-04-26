import argparse

import yaml
from typing import TypeAlias

MainOptions: TypeAlias = dict
LoggerOptions: TypeAlias = dict 


def get_options() -> tuple[MainOptions, LoggerOptions]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-options', type=str, required=True, help='Path to options file.')
    parser.add_argument('-logger_options', type=str, required=True, help='Path to logger option\'s file')
    args = parser.parse_args()

    option_path = args.options
    logger_path = args.logger_options
    with open(option_path, 'r') as project_options, open(logger_path, 'r') as logger_options:
        options = yaml.safe_load(project_options)
        logger_options = yaml.safe_load(logger_options.read())
    
    return (options, logger_options)
