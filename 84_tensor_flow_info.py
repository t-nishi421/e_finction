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

''' 色々
【行列計算】
足し算          : tf.add(a, b)
引き算          : tf.subtract(a, b)
掛け算          : tf.matmul(a, b)
成分同士の掛け算 : tf.multiply(a, b)

Σ   : tf.reduce_sum()
log : tf.log()

SGDで最適化 : tf.train.GradientDescentOptimizer(学習率)).minimize(最小化したい関数) 

予測結果と正解ラベルを比較 : tf.equal()

variableを初期化 : tf.global_variables_initializer() 
'''

''' データ加工
ランダムクロップ    : tf.random_crop(画像, [縦幅, 横幅, 謎の数値])
ランダム輝度        : tf.image.random_brightness(画像, max_delta, seed=None)
ランダムコントラスト : tf.image.random_contrast(画像, 下限値, 上限値, seed=None)
'''
