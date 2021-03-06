import os
import logging
import logging.handlers


def _get_logger(logdir, logname, loglevel=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt)

    handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(logdir, logname),
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger("")
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    return logger


def _set_basic_logging():
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
