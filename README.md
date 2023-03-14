# 2023-A32-WorkList

## What's in 'utils.py'
GetDataset: to integrate input data and ouput data

load_array: input a integrated dataset with the format of (input, output) and you will get a data iterator

normalization: input a list to get (max number, min number, a standarized list), the list is reflected to [-1, 1]

ArrNorm: use function 'normalization' to standarize a np.ndarray type data

df2arr: transfer list to np.ndarray with dtype=np.float32

R_square: calculate R-squared criteria to evaluate input (predict, ground truth)