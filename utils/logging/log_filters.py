import logging


class ErrorFilter:
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelname in ['ERROR', 'CRITICAL']


class InfoFilter:
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelname == 'INFO'


class TelegramTrueFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.telegram:
            return True


class TelegramFalseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.telegram:
            return True
    
