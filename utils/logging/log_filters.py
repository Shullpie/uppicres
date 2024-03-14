import logging


class TelegramInfoFilter(logging.Filter):
    def filter(self, record):
        return record.msg.startswith('🔴')


class TelegramChartFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.msg.endswith('chart.png')


class StdoutInfoFilter(logging.Filter):
    def filter(self, record):
        return 'chart.png' not in record.msg
    
