# 線形回帰で用いる関数
import numpy as np

def add_ones_to_firstcolum(X):
    """ # 行列の1列目に「1」を追加する

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
        - y (numpy.ndarray): 実測値列ベクトル size:m

    Returns:
        - numpy.ndarray: 目的関数J(θ)を最小にするθ
    """
    return np.linalg.pinv(X.T@X)@X.T@y

def computeCost(h, y):
    """ # 目的関数Jの計算を行う
    J = 1/2m Σ,m,i=1,(h(x(i)) - y(i))**2

    Args:
        h (numpy.ndarray): 仮説ベクトル
        y (numpy.ndarray): 実数値ベクトル

    Returns:
        numpy.float64: J
    """
    m = y.size
    return 1/(2*m) * np.sum((h - y)**2)
