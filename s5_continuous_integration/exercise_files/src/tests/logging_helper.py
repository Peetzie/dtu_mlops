import logging
import logging.config
import sys
from pathlib import Path
import os
from rich.logging import RichHandler


class LoggerConfigurator:
    def __init__(self, folder_name):
        self.logs_dir = os.path.join('log', folder_name)
        # Convert the string path to a Path object
        self.logs_dir = Path(self.logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._configure_logging()

    def _configure_logging(self, level=logging.DEBUG):
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'minimal': {'format': '%(message)s'},
                'detailed': {
                    'format': '%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n'
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'stream': sys.stdout,
                    'formatter': 'minimal',
                    'level': logging.INFO,
                },
                'info': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': Path(self.logs_dir, 'info.log'),
                    'maxBytes': 10485760,  # 10 MB
                    'backupCount': 10,
                    'formatter': 'detailed',
                    'level': logging.INFO,
                },
                'error': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': Path(self.logs_dir, 'error.log'),
                    'maxBytes': 10485760,  # 10 MB
                    'backupCount': 10,
                    'formatter': 'detailed',
                    'level': logging.ERROR,
                },
            },
            'root': {
                'handlers': ['console', 'info', 'error'],
                'level': logging.INFO,
                'propagate': True,
            },
        }

        logging.config.dictConfig(logging_config)
        logger = logging.getLogger()
        logger.handlers[0] = RichHandler(markup=True)  # set rich handler

    def get_logger(self):
        return logging.getLogger()
