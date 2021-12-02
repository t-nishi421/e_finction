# ニューラルネットワーク基礎

''' 多クラスロジスティック回帰
2種より多くの分類が可能（単純ロジスティック回帰は２種分類が限界）
入力と出力が複数あるもの
'''

''' ソフトマックス関数
要するに確率に換算する
'''

''' ネットワークの学習
手順１：順伝播でモデルの出力を求める
手順２：コスト関数で誤差を評価
手順３：逆伝播で勾配を求める
手順４：求めた勾配を基にパラメータを更新する
'''

''' 局所最小値を回避する手法
【確率的勾配降下法】
一部のデータを用いて学習する手法

【例】
エポック × ミニバッチ処理
一部のサンプルを抽出（ミニバッチ）し学習を行う。
これを複数回行う。（繰り返し回数 = エポック数）
'''

''' お勧めのパラメータ初期化
【重み】
- 0に近いランダムな値
- 平均0、分散1の正規分布
np.random.uniform(low=-1.0, high=1.0, size=(shape))
np.random.randn(shape=size)

- その他の手法（実装例は割愛）
LeCun初期化
Glorot初期化
He初期化

【バイアス】
- バイアスは0等の定数初期化がよく利用される
'''

import numpy as np

def cross_entropy_function(y_hat, y, m):
    """ クロスエントロピー関数

    Args:
        y_hat ([type]): モデルの出力
        y ([type]): 正解ラベル
        m ([type]): 訓練データの個数

    Returns:
        [type]: cost
    """
    return -np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

def multi_class_cross_entropy_function(y_hat, y, m):
    """ マルチクラスクロスエントロピー関数

    Args:
        y_hat ([type]): モデルの出力
        y ([type]): 正解ラベル
        m ([type]): 訓練データの個数

    Returns:
        [type]: cost
    """
    return -np.sum(y * np.log(y_hat)) / m

def soft_max(x):
    """ ソフトマックス関数 """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def step(x):
    """ ステップ関数 """
    return np.array(x > 0, dtype=np.int)

def relu(x):
    """ ReLu関数 """
    return np.muximum(0, x)

#######################
# ミニバッチ勾配降下法 #
#######################
''' sampleデータ
X           = np.arrange(1, 16, 1)
epochs      = 3
sample_size = 15
batch_size  = 5
'''
def mini_batch(X, epochs, sample_size, batch_size):
    """ ミニバッチ勾配降下法 """
    for _ in range(epochs):
        for j in range(0, sample_size, batch_size):
            X_batch = X[j:j + batch_size]
            print(X_batch)
        
