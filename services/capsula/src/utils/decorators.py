import time
import logging
from logging.config import dictConfig

from logs.log_config import LogConfig

dict_config = LogConfig()
dictConfig(dict_config.model_dump())
logger = logging.getLogger(dict_config.LOGGER_NAME)


def fun_timer(function):
    def wrapper(*args, **kwargs):
        fun_name = function.__name__
        msg = f'Iniciando llamada a función {fun_name}'
        logger.info(msg)
        ini_time = time.time()
        function(*args, **kwargs)
        end_time = time.time() - ini_time
        msg = f'La función {fun_name} se ejecutó durante {end_time:.2f} segundos.'
        logger.info(msg)
    
    return wrapper
