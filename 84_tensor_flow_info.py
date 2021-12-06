# Tensor Flowについて

''' 全ての要素が0のテンソルの作成
zero = tf.zeros([row_dim, col_dim])
'''

''' 全ての要素が1のテンソルの作成
one = tf.ones([row_dim, col_dim])
'''

''' 任意を行列サイズに任意の定数のテンソルの作成
filled = tf.fill([row_dim, col_dim], 定数)
'''

''' 任意の定数でテンソルの作成
cons = tf.constant([1,2,3])
'''

''' L2正則化
# パラメータ更新時に正則化を導入
# 一般的には過学習を抑制出来る
w = (1 - αλ)w - α ∂C/∂w
b = b - α ∂C/∂b
'''