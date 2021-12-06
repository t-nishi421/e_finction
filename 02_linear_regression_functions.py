# 線形回帰
import numpy as np
import matplotlib.pyplot as plt

def x_with_bias(X):
    """ # 行列の1列目に「1」を追加する

    Args:
        X (numpy.ndarray): 行列X

    Returns:
        numpy.ndarray: 各行の1列目に「1」が追加された行列X
    """
    m = X.size
    return np.c_[np.ones([m, 1]), X]

def normal_equation(X, y):
    """ # 正規方程式
    目的関数J(θ)を最小にするθを出力
    m:= データの個数（行）
    n:= 特徴の個数（列）

    【数式】
    θ = (Xt X)-1 Xt y

    Args:
        - X (numpy.ndarray): 計画行列 size:m(n+1)
        - y (numpy.ndarray): 実測値列ベクトル size:m

    Returns:
        - numpy.ndarray: 目的関数J(θ)を最小にするθ
    """
    return np.linalg.pinv(X.T@X)@X.T@y

def computeCost(h, y):
    """ # 目的関数Jの計算を行う
    【数式】
    J = 1/2m Σ,m,i=1,(h(x(i)) - y(i))**2

    Args:
        h (numpy.ndarray): 仮説ベクトル
        y (numpy.ndarray): 実数値ベクトル

    Returns:
        numpy.float64: J
    """
    m = y.size
    return 1/(2*m) * np.sum((h - y)**2)

def the_steepest_descent(X, y, alpha=0.01, epochs=10000, learn_plt=False):
    """# 短回帰の最小勾配法
    m := 訓練例の数
    h = ax + b := 仮説

    Args:
        X (numpy.ndarray): 計画行列
        y (numpy.ndarray): 実測値列ベクトル
        alpha (float, optional): 学習率. Defaults to 0.01.
        epochs (int, optional): 学習回数. Defaults to 10000.
        learn_plt (bool, optional): 学習のプロット. Defaults to False.

    Returns:
        numpy.ndarray: 二乗和誤差が最小になる係数、定数
    """
    m = X.size
    cost = []

    a, b = 0, 0
    θ = np.array([[0.], [0.]])

    for _ in range(epochs):
        h = a*X + b
        cost.append(1 / (2*m) * np.sum((h - y)**2))

        a = a - alpha/m * np.sum((h - y)*X)
        b = b - alpha/m * np.sum(h - y)

    if learn_plt:
        # 序盤の学習は視認性を考慮して省いている
        plt.plot(cost[10:])

    return np.array([a, b])

def computeCost(XwithBias, y, theta):
    """# 二乗和誤差が最小になるJを計算
    【数式】
    J(θ0, θ1) = 1/2m { (XwithBias・θ - y)**2 }

    Args:
        XwithBias (numpy.ndarray): 左に１を追加した行列
        y (numpy.ndarray): 実数値列ベクトル
        theta (numpy.ndarray): 更新するパラメータ

    Returns:
        numpy.float64: J
    """
    m = y.size
    J = 1 / (2*m) * np.sum((XwithBias@theta - y)**2)
    return J

def gradientDescent(XwithBias, y, theta, alpha, iterations):
    m = y.size
    J_history = np.empty([iterations, 1])
    
    for iter in range(iterations):
        theta = theta - (alpha/m)*(XwithBias.T@(XwithBias@theta - y))
        J_history[iter] = computeCost(XwithBias, y, theta)

    return (theta, J_history)
