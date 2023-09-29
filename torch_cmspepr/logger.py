import logging

def setup_logger(name: str = "cmspepr") -> logging.Logger:
    """Sets up a Logger instance.

    If a logger with `name` already exists, returns the existing logger.

    Args:
        name (str, optional): Name of the logger. Defaults to "demognn".

    Returns:
        logging.Logger: Logger object.
    """
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
    else:
        fmt = logging.Formatter(
            fmt=(
                "\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m"
                " %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger

logger = setup_logger()