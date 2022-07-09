import os
import random
import numpy as np
import torch

def set_seed(seed):
    """ fixed seed """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device):
    """ set-up device """
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    cuda = not cpu and torch.cuda.is_available()

    return torch.device('cuda:0' if cuda else 'cpu')



