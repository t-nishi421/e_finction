# NNの改善

import numpy as np

def tanh(x):
    """ ハイパボリックタンジェント関数 """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    """ ハイパボリックタンジェント関数の微分 """
    return 4 / (np.exp(x) + np.exp(-x))**2

def leaky_relu(x):
    """ LeakyReLU関数 """
    return np.maximum(0.01*x, x)

def leaky_relu_derivative(x):
    """ LeakyReLU関数の微分 """
    return [1 if i > 0 else 0.01*i for i in x]

