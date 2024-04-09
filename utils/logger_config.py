# coding=utf-8
from loguru import logger

def setup_logger():
    logger.add("error.log", rotation="10 MB")
    return logger

my_logger = setup_logger()
