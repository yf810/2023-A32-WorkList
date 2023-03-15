import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import lr_scheduler
from torch.optim import lr_scheduler
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def GetDataset(input_arr: list, output_arr: list, seq: int):
    assert(len(input_arr)==len(output_arr)), "Different size of input and output!"
    Input = []
    Output = []
    for i in range(input_arr.shape[0]-seq):
        Input.append(input_arr[i:i+seq][:])
        Output.append(output_arr[i:i+seq][:])
    return torch.tensor(Input, dtype=torch.float32), torch.tensor(Output, dtype=torch.float32)

        
def load_array(data_arrays, batch_size, is_train=True):
    # data-iter
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def normalization(x: list):
    M, m = np.max(x), np.min(x)
    for i in range(len(x)):
        x[i] = (x[i] - (M + m) / 2) / ((M - m) / 2)
    # x in [-1, 1]
    return M, m, x

def ArrNorm(x: np.ndarray):
    assert isinstance(x, np.ndarray), "We need a list"
    M_list, m_list, res = [], [], []
    for i in range(x.shape[0]):
        u = x[i].tolist()
        M, m, t = normalization(u)
        res.append(t)
        M_list.append(M)
        m_list.append(m)
    return M_list, m_list, np.array(res)


def df2arr(x) -> np.ndarray:
    return np.array(x, dtype=np.float32)



def R_square(A: torch.tensor, B: torch.tensor) -> torch.float32:
    assert A.shape == B.shape, "Predict value not match the Ground Truth"
    # A: predict   B: ground truth
    # shape: batch_size * 1 * w * h
    A = A.detach()
    B = B.detach()
    A = A.squeeze()
    B = B.squeeze()
    flag = len(A.shape)==3
    # batch_size * w * h
    *_, h = A.shape
    gt_bar = torch.mean(B, dim=[0,1] if flag else 0, keepdim=False)

    def sq_sum(x):
        x = torch.tensor(x, dtype=torch.float32)
        return torch.sum(x * x, dim=[0,1] if flag else 0)
    
    SST = [sq_sum(A[:,:,i] if flag else A[:,i] - gt_bar[i]) for i in range(h)]
    SSR = [sq_sum(B[:,:,i] if flag else A[:,i] - gt_bar[i]) for i in range(h)]

    return [ (SST[i] / SSR[i]) for i in range(h) ]

"""
R-squared = SSR / SST = 1 - SSE / SST
"""
