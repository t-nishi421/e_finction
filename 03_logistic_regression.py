# ロジスティック回帰
import numpy as np

"""[summary]
【仮説】
h = 1 / (1 + e^-θ.T@X)
※ 「-θ.T@X」の部分は場合に追って計算可能な状態に変更する必要あり。

【目的関数】
J = (-1/m) * Σ( y@log(h) + (1-y)@log(1-h) )

【勾配】
grad = 1/m * Σ(h-y)@X

"""

def sigmoid(X, θ):
	z = θ[0] + θ[1]*X
	h = 1 / (1 + np.exp(-z))
	return h

def computeCost_logi(h, y):
	"""# ロジスティック関数の目的関数J(θ)を計算する関数

	Args:
		h (numpy.ndarray)): 仮説
		y (numpy.ndarray): 実数値

	Returns:
		numpy.float64: J
	"""
	m = y.shape[0]
	J = -(1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
	return J

def hypothesis(X, theta):
    """# 仮説 h(θ) を計算する関数

    Args:
        X (numpy.ndarray): 計画行列
        theta (numpy.ndarray): 更新するパラメータ

    Returns:
        numpy.ndarray: 仮説
    """
    m = X.shape[0] #訓練例の数
    XwithBias = np.c_[np.ones([m,1]),X] # Xの1列目にズラッと1を並べる XwithBias.shapeは(100, 3)
    #numpyを使ってもscipyを使っても構いません。
    h = 1 / (1 + np.exp(-XwithBias@theta))
    return h

def predict(theta, X):
    """# 訓練集合に対する予測精度を計算

    使用例
    predictX = predict(theta, X)
    accuracy = (predictX == y).mean() # 予測predictXと正解yが一致しているデータ
    print("学習精度は %2.1f"% (accuracy*100), "%")

    Args:
        theta ([type]): [description]
        X ([type]): [description]

    Returns:
        [type]: [description]
    """
    h = hypothesis(X, theta)
    predicted_y = np.where(h >= 0.5, 1, 0 )
    return predicted_y

def polyFeature(X0, X1, degree):
    X0 = X0[:, np.newaxis] #np.appendで付け足していくため、2-dimにする
    X1 = X1[:, np.newaxis]
    Xpoly = np.empty((X0.shape[0], 0)) # m行0列の配列を作成。これに新たな特徴を付け足していく。

    for i in range(1, degree+1):
        for j in range(i+1):
            Xpoly = np.append(Xpoly, ((X0**(i-j)) * (X1**j)), axis=1)

    return Xpoly

def cost(theta, X, y):
    """# 目的関数Jを計算する関数

    Args:
        theta ([type]): [description]
        X ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    h = hypothesis(X, theta) #仮説h
    m = X.shape[0] #訓練例の数

    # 目的関数J(θ)の実装
    # 1次元のnumpy配列同士のdot積は内積を表します。
    J = (-1/m)*np.sum(y@np.log(h) + (1-y)@np.log(1-h))

    # -inf * 0 は nan になる。そのときはJ全体をinfで返す。
    if np.isnan(J):
        return np.inf
    return J

def grad(theta, X, y):
    """# 目的関数の勾配を計算する関数

    Args:
        theta ([type]): [description]
        X ([type]): [description]
        y ([type]): [description]
    """
    m = X.shape[0] #訓練例の数: 100
    XwithBias = np.c_[np.ones([m,1]),X] # Xの1列目にズラッと1を並べる XwithBias.shapeは(100, 3)
    h = hypothesis(X, theta) #仮説h

    # ヒント：XwithBiasを使います。この行列の転置はXwithBias.T
#     grad = (1/m) * np.sum(XwithBias.T@(h-y))
    grad = XwithBias.T@(h-y) / m
    return grad

"""良く分らんやつ

#01 多項式で特徴を増やす
プロットしたグラフを見たら分かるように、このデータは直線ではy=0とy=1を分離できなさそうです。多項式で特徴を増やしましょう。
def polyFeature(X0, X1, degree):
    X0 = X0[:, np.newaxis] #np.appendで付け足していくため、2-dimにする
    X1 = X1[:, np.newaxis]
    Xpoly = np.empty((X0.shape[0], 0)) # m行0列の配列を作成。これに新たな特徴を付け足していく。

    for i in range(1, degree+1):
        for j in range(i+1):
            Xpoly = np.append(Xpoly, ((X0**(i-j)) * (X1**j)), axis=1)

    return Xpoly

#02 決定境界
決定境界を引く関数plotDataWithDB2(X, y, theta, deg)を作ります。今回は複雑な曲線であるため、これまでのように端と端を計算してplotし直線を引くという手法が使えません。どうすればよいでしょうか。

ここで使う手法はよく使われるので、是非コードを熟読して下さい。
def plotDataWithDB2(X, y, theta, deg):
    #課題１で実装してもらった関数です
    plotData2(X,y)
    
    #このセル内で直前にプロットしたものに上書きしていきます
    plt.hold(True)
    
    #横軸も縦軸も0から3まで、均等に50サンプル取ってきます。
    X0_lin = np.linspace(0, 3, 50)
    X1_lin = np.linspace(0, 3, 50)
    
    #そしてこの50×50=2500個の地点におけるx(バイアス込み)とthetaの内積をそれぞれ計算してきます。
    y_plot = np.empty((X0_lin.size, X1_lin.size))
    
    for i in range(X0_lin.size):
        for j in range(X1_lin.size):
            y_plot[i, j] = np.c_[[1], polyFeature( np.array([X0_lin[i]]), np.array([X1_lin[j]]), deg)].dot(theta)
    
    #この2500個のサンプルに対して、等高線を引きます。値がゼロのところが決定境界ですので、levels=[0]のみ等高線を引きます。
    plt.contour(X0_lin, X1_lin, y_plot.T, levels=[0])
    
    #念のため、グラフのレンジをもう一度指定しておきます。
    plt.xlim([0,3])
    plt.ylim([0,3])
    
    plt.show()

"""

""" L2正則化について
以外と知らない「最小二乗法」の落とし穴～正則化項を使おうよ！～
<https://www.youtube.com/watch?v=KNE-BUKGyDk>

【数学嫌いと学ぶデータサイエンス】第6章-第2回-ridge,lasso回帰
<https://www.youtube.com/watch?v=noNlFZD7Cbw&t=22s>
"""

#正則化された目的関数
def costReg(theta, X, y, lmd):
    m = y.size
    J = cost(theta, X, y)

    J = J + lmd * np.square(np.linalg.norm(theta[1:])) / (2 * m)

    return J

#正則化された目的関数の勾配を計算する関数
def gradReg(theta, X, y, lmd):
    m = y.size
    g = grad(theta, X, y)
    theta[0] = 0 #0番目の成分は正則化項に含めないため、0にしておきます。
    
    g = g + ((lmd/m) * theta)

    return g

