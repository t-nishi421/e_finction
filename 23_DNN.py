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

''' 半教師あり学習の事例
VAE
Variational Auto Encoder
変分自己符号化法
Ladder Network
'''

''' モメンタム
学習を高速化する手法の一つ
パラメータ更新時にモメンタム項を加える
'''

''' Adagrad
- 学習率調整アルゴリズム
- 過去の勾配の二乗和の平方根に反比例するようにスケーリングして学習率を適応させる手法。

α: 学習率
θ: パラメータ
δ: 小さい定数10^-7
h=0: 勾配の類型を蓄積する変数

【ステップ１】勾配を計算
∂C/∂θt
【ステップ２】hに勾配の二乗を蓄積する
h = h + ∂C/∂θt ⦿ ∂C/∂θt
【ステップ３】除算と平方根を要素ごとに適用し、パラメータを更新
θt+1 = θt - α/δ+√h ∂C/∂θt

- あらゆるモデルで上手くいくわけではないので、過信は禁物
'''

''' RMSprop
Adagradの改良版
'''

''' Adam
適応的学習率最適化アルゴリズム
RMSpropとモメンタムの組み合わせ的な奴
'''

''' ニュートン法
https://www.youtube.com/watch?v=4sjfMvuitnE
'''

''' 言葉だけ覚えておいた方が良いやつ
座標降下法
ポルヤック平均化
教師あり事前学習
'''
