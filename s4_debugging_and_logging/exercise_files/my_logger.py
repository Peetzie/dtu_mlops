import logging
import logging.config
import sys
from pathlib import Path
from rich.logging import RichHandler


# Define the LOGS_DIR path
LOGS_DIR = Path('log/my_logger/')  # Ensure this directory exists or is created
LOGS_DIR.mkdir(exist_ok=True)

# Define the logging configuration
logging_config = {
    'version': 1,
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
            'level': logging.DEBUG,
        },
        'info': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(
                Path(LOGS_DIR, 'info.log')
            ),  # Convert Path object to string
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 10,
            'formatter': 'detailed',
            'level': logging.INFO,
        },
        'error': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(
                Path(LOGS_DIR, 'error.log')
            ),  # Convert Path object to string
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 10,
            'formatter': 'detailed',
            'level': logging.ERROR,
        },
    },
    'root': {
        'handlers': ['console', 'info', 'error'],
        'level': logging.DEBUG,
        'propagate': True,
    },
}

# Apply the logging configuration
logging.config.dictConfig(logging_config)

# Create and use logger
logger = logging.getLogger(__name__)

# Add coloring

logger.root.handlers[0] = RichHandler(markup=True)  # set rich handler

# Logging levels (from lowest to highest priority)
logger.debug('Used for debugging your code.')
logger.info('Informative messages from your code.')
logger.warning('Everything works but there is something to be aware of.')
logger.error("There's been a mistake with the process.")
logger.critical('There is something terribly wrong and process may terminate.')
