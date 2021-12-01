# np.newaxisとは何か？
Python/Numpyのnp.newaxisの考え方のコツ
<https://qiita.com/rtok/items/10f803a226892a760d75>

# 平均値を求めたい np.mean()
```python
import numpy as np

ary = np.arange(12).reshape(3,4)
ary.mean(axis=None)

''' axisについて
axis=x
- None : ndarray全体の平均
- 0    : 列ごとの平均
- 1    : 行ごとの平均
'''
```

# 標準偏差を求めたい np.std()
```python
import numpy as np

ary = np.arange(12).reshape(3,4)
ary.std(axis=None)

''' axisについて
なんか結構めんどくさい構造になってる。。。
None, 0, 1は大体他と考え方は同じ
'''
```

# 配列をシャッフルしたい np.random.premutation(配列)
```python
import numpy as np

# 行ベクトル(n=100)を生成
X = np.array([i for i in range(100)])

np.random.permutation(X)
```

# 行列のアンロールと復元
```python
# flatten()を使用すると行列を行ベクトルに変換できる
# W = np.ndarray行列(row=10, colmun=11)
Wvec = W.flatten()

# 元に戻すときはreshape({行数}, {列数}) を使用する
W = Wvec.reshape(10, 11)
```