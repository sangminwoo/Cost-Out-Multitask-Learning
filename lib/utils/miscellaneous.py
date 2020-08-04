from datetime import datetime

def get_timestamp():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    st = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H:%M:%S')
    return st

def save_config(cfg, path):
    # if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())