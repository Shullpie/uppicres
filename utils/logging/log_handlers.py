import logging
import requests
from PIL import Image

TELEGRAM_CHAT_ID = 858660939
TELEGRAM_TOKEN = '6805704720:AAGfNoyqIWE2X39ydhmWZ1V3ssAUTIIHtQk'


class TelegramInfoHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': log_entry,
            'parse_mode': 'Markdown'
        }
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        return requests.post(url, data=payload)


class TelegramChartHandler(logging.Handler):
    def emit(self, record):
        caption, path = self.format(record).split('%')
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': caption,
            'parse_mode': 'Markdown'
        }
        with open(path, 'rb') as img:
            url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'
            return requests.post(url, data=payload, files={'photo': img})
