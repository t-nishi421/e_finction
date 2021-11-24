# オーバーフィッティング抑制

""" 正則化の学習に踏み込みたい方へ
L2正則化
- 目的関数にパラメータの二乗和を付け加えることで行う正則化
- L1空間、L2空間、Ln空間等がある。Lはルベーグのこと
チコノフ正則化
- パラメータごとに重みを変えることが出来る正則化
リッジ回帰
- L2正則化された線形回帰のこと
スパース行列
- 要素がほとんど0の行列
- L1正則化を行うとスパース行列になる
- どこが0になっていないかを調べれば入力するデータの構造の分析もできる
L1正則化
- スパース性を高める
Lasso回帰
- L1正則化を施した線形回帰
Elastic Net
- Lasso回帰とリッジ回帰を組み合わせたもの
"""

""" スケーリング
スケールの異なるデータを加工して整えること。
スケーリングすることで、勾配降下法の学習効率が向上する。

# Mean Normalization
xiを平均がだいたい0になるようにxi-μiと置き換える。

μi:= 訓練集合全体のxiの平均
si:= 訓練集合全体のxiの標準偏差（とりあえず最大値-最小値でおｋ）

xi:= xi-μi / si
"""

""" 演習
訓練集合：X, y
交差検証集合：Xcv, ycv
テスト集合：Xtest, ytest
"""
import numpy as np

def linearRegCostFunction(theta, XwithBias, y, lmd):
    """# 正規化された線形回帰の目的関数

    【計算式】
    J = 1/2m Σ(hθ(x(i)) - y(i))^2 + λ/2m Σθj^2

    Args:
        theta (np.ndarray): 更新するパラメータ
        XwithBias (np.ndarray): 計画行列
        y (np.ndarray): 行ベクトル
        lmd (int): 正則化パラメータ

    Returns:
        np.float64: 目的関数出力
    """
    m = y.size # データ数m
    J = (1/(2*m)) * np.sum((XwithBias@theta - y)**2) + (lmd/(2*m)) * np.sum(theta[1:]**2)
    return J

def linearRegGrad(theta, XwithBias, y, lmd):
    """# 正則化された線形回帰の目的関数の勾配

    Args:
        theta (np.ndarray): 更新するパラメータ
        XwithBias (np.ndarray): 計画行列
        y (np.ndarray): 行ベクトル
        lmd (int): 正則化パラメータ

    Returns:
        np.ndarray: 勾配
    """
    m = y.size # データ数
    grad = XwithBias.T@(XwithBias@theta - y) / m + (lmd/m) * np.r_[0, theta[1:]]
    return grad

import scipy.optimize as scopt
from functools import partial

def trainLinearReg(XwithBias, y, lmd):
    """# 線形回帰モデルを使って学習する関数

    Args:
        XwithBias (np.ndarray): 計画行列
        y (np.ndarray): 行ベクトル
        lmd (int): 正則化パラメータ

    Returns:
        anp.ndarray: 最適化されたパラメータ
    """
    _, n = XwithBias.shape # Xはここではバイアス項を付け加えているので2-dim

    # thetaの初期化
    #initial_thetaは1-dim
    initial_theta = np.zeros(n)

    # 最適化する変数以外は固定
    # theta以外固定したcost function
    cost_fixed = partial(linearRegCostFunction, XwithBias=XwithBias, y=y, lmd=lmd) #functools
    # theta以外固定したgraf function
    grad_fixed = partial(linearRegGrad, XwithBias=XwithBias, y=y, lmd=lmd) #functools

    # 最適化
    res = scopt.minimize(cost_fixed, initial_theta, jac=grad_fixed, method='BFGS')

    return res.x

def learningCurve(XwithBias, y, XcvWithBias, ycv, lmd):
    """# 学習曲線をプロットするための、学習誤差と交差検証誤差を出力する関数
    
    データの個数増加に伴う予測精度の向上を観測する。

    Args:
        XwithBias (np.ndarray): 学習計画行列
        y (np.ndarray): 学習行ベクトル
        XcvWithBias (np.ndarray): 交差検証計画行列
        ycv (np.ndarray): 交差検証行ベクトル
        lmd (int): 正則化パラメータ

    Returns:
        np.ndarray: 学習誤差 error_traion
        np.ndarray: 交差検証誤差 error_cv
    """
    m, _ = XwithBias.shape # 2-dim
    m_cv, _ = XcvWithBias.shape

    # np.empty(配列要素数)は、初期化をせずに新しい配列を生成する際に使用します。np.onesなどに比べ、初期化の処理がないため高速です。
    error_train = np.empty(m)
    error_cv = np.empty(m)

    for i in range(m):
        theta = trainLinearReg(XwithBias[:i+1], y[:i+1], lmd)
        error_train[i] = (1 / (2*(i+1))) * np.sum((XwithBias[:i+1]@theta - y[:i+1])**2)
        error_cv[i] = (1 / (2*(m_cv))) * np.sum((XcvWithBias@theta - ycv)**2)
        
    return [error_train, error_cv]