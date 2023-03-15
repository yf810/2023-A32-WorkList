
import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.utils import data
from sklearn.preprocessing import MinMaxScaler

Epoch_num = 200
Learning_Rate = 0.001

class mlp(nn.Module):
    
    def __init__(self, d_in, d_hidden, d_out):
        
        super(mlp, self).__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(d_hidden, d_out),
            torch.nn.Tanh()
        )
        
        #self.mlp.weight.data = get_weight_initial(d_out, d_in)
        
    def forward(self, x):
        H_out = self.mlp(x)
        return H_out
    
def normalization1(x):
    x = x.detach().numpy()
    scalar = MinMaxScaler(feature_range=(0, 1))
    x_nor = scalar.fit_transform(x)
    return x_nor

def df2arr(x) -> np.ndarray:
    return np.array(x, dtype=np.float32)

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

excel = pd.read_excel('A32.xlsx', header=None)
sp = [1486, 2972, 4458]
station_1 = excel.iloc[1:sp[0]+1,1:6]
station_2 = excel.iloc[sp[0]+1:sp[1]+1,1:6]
standard = excel.iloc[sp[1]+1:sp[2]+1,1:6]
station_1 = df2arr(station_1)
station_2 = df2arr(station_2)
standard = df2arr(standard)

station_1 = torch.from_numpy(station_1)
station_2 = torch.from_numpy(station_2)
standard = torch.from_numpy(standard)

def loss_func(pred1, labels):
    mse = torch.nn.MSELoss()
    Loss= mse(pred1, labels)
    return Loss

def R_square(A: torch.tensor, B: torch.tensor) -> torch.float32:
    assert A.shape == B.shape, "Predict value not match the Ground Truth"
    # A: predict   B: ground truth
    # shape: batch_size * 1 * w * h
    A = A.detach()
    B = B.detach()
    A = A.squeeze()
    B = B.squeeze()
    # batch_size * w * h
    *_, h = A.shape
    pre_bar = torch.mean(A, dim=[0,1], keepdim=False)
    gt_bar = torch.mean(B, dim=[0,1], keepdim=False)
    # print(pre_bar.shape[0])

    def sq_sum(x):
        # print(x.shape)
        x = torch.tensor(x, dtype=torch.float32)
        return torch.sum(x * x, dim=[0,1])
    # print(A[:,:,1].shape, pre_bar[1].shape)
    SST = [sq_sum(A[:,:,i] - gt_bar[i]) for i in range(h)]
    SSR = [sq_sum(B[:,:,i] - gt_bar[i]) for i in range(h)]


    return [ (SST[i] / SSR[i]) for i in range(h) ]

model = mlp(d_in = station_1.shape[1], d_hidden = 256, d_out = standard.shape[1])
optimzer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)

for epoch in range(Epoch_num):
    pred = model(station_1)
    loss = loss_func(pred, standard)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    #print("Epoch: [{}]/[{}]".format(epoch + 1, Epoch_num))
    r = R_square(pred, standard)
    print("Epoch {}, R {}".format(epoch, r))

