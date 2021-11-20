# メモ帳
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

def add_ones_to_firstcolum(X):
    """ # 各ベクトルの1列目に「1」を追加する

    Args:
        X (numpy.ndarray): 行列X

    Returns:
        numpy.ndarray: 各行の1列目に「1」が追加された行列X
    """
    X_row, _ = X.shape
    return np.c_[np.ones([X_row, 1]), X]

def normal_equation(X, y):
    """ # 正規方程式
    目的関数J(θ)を最小にするθを出力
    m:= データの個数（行）
    n:= 特徴の個数（列）

    Args:
        - X (numpy.ndarray): 計画行列 size:m(n+1)
        - y (numpy.ndarray): 実数値列ベクトル size:m

    Returns:
        - numpy.ndarray: 目的関数J(θ)を最小にするθ
    """
    return np.linalg.pinv(X.T@X)@X.T@y

X = np.array([[1,2],[1,4]])
y = np.array([[1],[4]])
normal_equation(X, y)

def the_steepest_descent(X, y, alpha=0.01, epochs=10000):
    m = X.size
    cost = []

    # 更新パラメータ
    a = 0
    b = 0

    for _ in range(epochs):
        h = a*X + b
        cost.append(1 / (2*m) * np.sum((h - y)**2))

        a = a - alpha/m * np.sum((h - y)*X)
        b = b - alpha/m * np.sum(h - y)

    print(f'a:{a}')
    print(f'b:{b}')
    return a, b