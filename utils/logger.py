import logging
import os
import sys

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

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

    # for i in range(10):
    #     logger.info(f'{i}번째 방문입니다.')