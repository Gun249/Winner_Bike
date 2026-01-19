import logging


def setup_logger():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def use_logger():
    logger = logging.getLogger(__name__)
    return logger

setup_logger()
logger = use_logger()
