import torch
import numpy as np
import cv2

def read_from_csv_2f(path):
    """
    :param path: path to csv file containing data with 2 features
    :return: x1, x2, y (all are lists of floats)
    """
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    lines.pop(0) #remove header
    x1 = []
    x2 = []
    y = []
    for line in lines:
        _x1, _x2, _y = line.split(',')
        x1.append(float(_x1))
        x2.append(float(_x2))
        y.append(float(_y))
    return x1, x2, y

def scale(xi):
    """
    :param xi: np (n)
    :return: float, float, np (n)
    """
    _m = np.mean(xi)
    _s = np.max(xi)-np.min(xi)
    return _m, _s, (xi-_m)/_s

def img2tensor(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return torch.tensor(im, dtype=torch.float32)/255