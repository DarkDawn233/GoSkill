import random
import os
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_host_ip():
    """
    Get the host IP address.
    """
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def get_server_name():
    """
    Get the server name with IP address.
    """
    ip = get_host_ip()
    server_name = "s" + ip.split(".")[-1]
    return server_name


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'
