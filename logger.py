import logging
import os
import sys

def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # DEBUG, INFO, ERROR, WARNING

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    # stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(save_dir, filename))
    # file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger