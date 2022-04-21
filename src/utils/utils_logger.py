import logging


def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)02d: %(message)s', datefmt='%y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
