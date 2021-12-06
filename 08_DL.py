# Deep Learning
'''
絶対に理解させる誤差逆伝播法【深層学習】
https://www.youtube.com/watch?v=0itH0iDO8BE
'''

''' 勾配法
【決定版】スーパーわかりやすい最適化アルゴリズム -損失関数からAdamとニュートン法-
https://qiita.com/omiita/items/1735c1d048fe5f611f80

- バッチ勾配法
  どの繰り返しステップでも全ての訓練例を使う方法

- 確率的勾配法降下法
  各ステップでランダムにいくつかの訓練例を選んで使う勾配法

- 共役勾配法
  https://qiita.com/Dason08/items/27559e192a6a977dd5e5

- BFGS法

- Adam法
  https://arxiv.org/pdf/1412.6980.pdf

などなど
'''

''' DLの応用
- 画風変換
  The Deep Forger

- シーン認識

- ノイズ除去

- 音声認識

- 聞いている音楽からお勧め表示

- 手書き文字の生成

- ゲームのルールを自ら分析・プレイ

- pix2pix
  ラベルから写真のような絵にする
  モノクロからカラーにする
  航空写真から地図風にする
  昼から夜にする
  縁だけから写真のような絵にする

- 読唇術
  LipNet
'''
import numpy as np

def sigmoid(x):
  """ シグモイド関数 """
  return 1 / (1 + np.exp(x))

def sigmoid_derivative(x):
  """ シグモイドの微分 """
  return sigmoid(x)*(1-sigmoid(x))

def relu(x):
  """ ReLu関数 """
  return np.maximum(0, x)

def relu_derivative(x):
  """ ReLuの微分 """
  return [1 if i > 0 else 0 for i in x]

'''
【np.sum(axis=int:) について】
https://deepage.net/features/numpy-axis.html

【np.sum(deepdims=bool:) について】
https://snowtree-injune.com/2020/05/03/keepdims-z006/#toc4
'''
def softmax(x):
  """ ソフトマックス関数"""
  exp_x = np.exp(x)
  return exp_x / np.sum(exp_x, axis=1, keepdims=True)

###########################
# 例：scikit-learnでDL実装 #
###########################
from sklearn.neural_network import MLPClassifier
x = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([1, 1, 0, 0])
X_train, X_test = x, x  #訓練データとテストデータ
y_train, y_test = y, y  #訓練データとテストデータの正解
  # インスタンス作成
mlp = MLPClassifier(hidden_layer_sizes=(2), # 隠れ層のユニット数
                    activation='relu',  # 隠れ層の活性化関数はReLU
                    max_iter=10000,  # 学習回数
                    alpha=0, # L2正則化パラメータ
                    solver='sgd',# 学習アルゴリズム（確率的勾配降下法）
                    verbose=0, # 学習ログの表示 0:非表示 1:表示 
                    learning_rate_init=0.01)# 学習率

# 作成したインスタンスを使って学習
# 引数に訓練データと正解ラベルを渡す
mlp.fit(X_train, y_train)

print("訓練スコア: %f" % mlp.score(X_train, y_train))
print("テストスコア: %f" % mlp.score(X_test, y_test))

# 重みとバイアスの確認
## 重み
mlp.coefs_[0]
## バイアス
mlp.intercepts_[0]

