from loguru import logger
from loguru._logger import Logger

def setup_logger() -> Logger:
    """
    Set up and return a configured logger using the loguru library.

    Returns:
        Logger: A Loguru Logger object with specified configuration.
    """
    logger.add("error.log", rotation="10 MB")
    return logger


my_logger = setup_logger()
