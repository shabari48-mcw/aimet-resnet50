import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_directory='log'):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    logger.handlers.clear()
    logger.propagate = False

    file_handler = RotatingFileHandler(
        os.path.join(log_directory, 'app.log'),
        maxBytes=10**6,
        backupCount=3
    )

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(lineno)d-%(funcName)s-%(pathname)s')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Initialize the logger
logger = setup_logging()
