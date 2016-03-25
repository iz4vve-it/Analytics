import imp
import time
import logging
import itertools
import sklearn.preprocessing as process


def get_logger(logger_name, logger_level=logging.INFO, file_handler=""):
    """
    Returns a logger instance with a determined formatter

    :param logger_name: name of the logger to be instantiated (best set as __name__)
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
    ch = logging.StreamHandler()
    ch.setLevel(logger_level)
    formatter = logging.Formatter('[%(asctime)s]: %(name)s - %(funcName)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # generates file_handler, if required
    if file_handler:
        fh = logging.FileHandler(file_handler)
        fh.setLevel(logger_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


logger = get_logger(__name__)


def normalize_dataframe(array):
    """
    Function to normalize in the interval (0,1) all numeric columns in
    a dataframe.

    :param array: dataframe to normalize
    :returns: normalized dataframe
    """
    mm = process.MinMaxScaler()
    return mm.fit_transform(array.astype(float))


def nwise(iterable, n):
    """Returns sublists of n elements from iterable"""
    return itertools.izip(
        *(itertools.islice(iterable, i, None)
          for i, iterable in enumerate(itertools.tee(iterable, n)))
    )


def timeit(func):
    """Decorator to time function execution"""
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()

        logger.info('Execution of %r completed: %2.5f sec' % (func.__name__, te-ts))
        return result

    return timed


def debug_call(func):
    """Decorator to provide details regarding a function call"""
    def call_details(*args, **kw):
        logger.info("Function %r calleds with arguments: %s %s" % (func.__name__, args, kw))
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()

        logger.info('Execution of %r completed: %2.5f sec' % (func.__name__, te-ts))
        return result
    return call_details
