"""
Generic tools for computation
"""
import imp
import time
import logging
import itertools
import sklearn.preprocessing as process


def get_logger(logger_name, logger_level=logging.INFO, file_handler=None):
    """
    Returns a logger instance with a determined formatter

    :param logger_name: name of the logger to be instantiated
    :type logger_name: str
    :param logger_level: level of logging
    :type logger_level: int (or logging.level)
    :param file_handler: name of a file to use as log file
    :type file_handler: str (valid file name)

    :return logger: logger instance
    :rtype: logging.logger
    """
    logging.shutdown()
    imp.reload(logging)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logger_level)
    formatter = logging.Formatter(
        '[%(asctime)s]: %(name)s - %(funcName)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # generates file_handler, if required
    if file_handler:
        file_handler = logging.FileHandler(file_handler)
        file_handler.setLevel(logger_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


LOGGER = get_logger(__name__)


def normalize_dataframe(array):
    """
    Function to normalize in the interval (0,1) all numeric columns in
    a dataframe.

    :param array: dataframe to normalize
    :returns: normalized dataframe
    """
    scaler = process.MinMaxScaler()
    return scaler.fit_transform(array.astype(float))


def nwise(iterable, sub_length):
    """
    Returns sublists of n elements from iterable
    :param iterable: a generic iterable
    :param sub_length: length of sub-lists to be generated
    :returns: a list of tuples
    """
    return itertools.izip(
        *(itertools.islice(element, i, None)
          for i, element in enumerate(itertools.tee(iterable, sub_length)))
    )


def timeit(func):
    """
    Decorator to time function execution
    :param func: a function
    :returns: a time-logged version of func
    """
    def timed(*args, **kw):
        """
        Timing decorator
        """
        start_time = time.time()
        result = func(*args, **kw)
        end_time = time.time()

        LOGGER.info('Execution of {} completed: {:2.5f} sec'.format(
            func.__name__, end_time - start_time)
        )
        return result

    return timed


def debug_call(func):
    """
    Decorator to provide details regarding a function call
    :param func: a function
    :returns: a call-logged version of func
    """
    def call_details(*args, **kw):
        """
        Call and timing decorator
        """
        LOGGER.info("Function {} calleds with arguments: {} {}".format(
            func.__name__, args, kw)
        )
        start_time = time.time()
        result = func(*args, **kw)
        end_time = time.time()

        LOGGER.info('Execution of {} completed: {:2.5f} sec'.format(
            func.__name__, end_time - start_time)
        )
        return result
    return call_details
