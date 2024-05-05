import os
import glob
import numpy as np


def get_casses(path):    
    data_path = []
    for path in glob.glob(path + '/*'):
        flag = path.split(os.path.sep)[-1].split("_")[-4]
        if flag == "2":
            data_path.append(path)        
    return data_path


def decay_schedule(epoch, lr):
    if (epoch % 5 == 0) and (epoch != 0):
        lr = lr * 0.8
    return lr