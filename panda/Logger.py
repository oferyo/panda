import logging

loggers = {}


def my_logger(name):
    global loggers

    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s::%(levelname)s::%(message)s",
                                      "%Y-%m-%d %H:%M:%S")

        fh = logging.FileHandler("panda.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.propagate = False
        loggers[name] = logger
        return logger
