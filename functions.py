import numpy as np

def least_squares(X, y):
    """ 最小二乗法（未完成）
    Returns:
        [type]: [description]
    """
    m = X.size
    theta = np.ones(X.size)
    h = X@theta

    J = 1/(2*m)*((h - y)**2).sum()
    return J

def normal_equation(X, y):
    """ # 正規方程式法
    目的関数J(θ)を最小にするθを出力

    Args:
        - X (numpy.ndarray): 計画行列 size:m(n+1)
        - y (numpy.ndarray): 目的変数列ベクトル size:m

    Returns:
        - numpy.ndarray: 目的関数J(θ)を最小にするθ
    """
    return np.linalg.pinv(X.T@X)@X.T@y

X = np.array([[1,2],[3,4]])
y = np.array([[1],[4]])
print(type(normal_equation(X, y)))