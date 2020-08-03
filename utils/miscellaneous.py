from datetime import datetime

def get_timestamp():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    st = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H:%M:%S')
    return st