import torch


def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))


describe(torch.Tensor(2, 3))
describe(torch.rand(2, 3))   # случайное равномерное распределение
describe(torch.randn(2, 3))  # случайное нормальное распределение

