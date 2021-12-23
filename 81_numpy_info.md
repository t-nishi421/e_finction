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

# ゼロパディング
```python
# numpy.padの使い方が分からなかったのでメモ
# https://qiita.com/horitaku1124/items/6ae979b21ddc7256b872

# np.pad(行列, 前後の文字埋め, 埋め込み方式)
```

# アインシュタイン縮約記法(einsum)
この記法を使うと計算時間の短縮になるとかならないとか
- docs  
 https://numpy.org/doc/stable/reference/generated/numpy.einsum.html  
- Qiita  
 https://qiita.com/d-takagishi/items/94e47ecd1abc54978b44  

# 累乗計算 np.power()
```python
# 2の3乗
np.power(2,3)
> 8
```

# 平方根 np.sqet()
```python
# √2（実数）
np.sqrt(2)
> 1.4142135623730951
```

# 複数のndarray配列を既存の軸に沿って結合する np.concatenate()
- 参考記事  
 https://note.nkmk.me/python-numpy-concatenate-stack-block/  
```python
# 2*3の行列を二つ作成
zeros = np.zeros([2,3])
print(f"zeros:\n{zeros}")
ones = np.ones([2,3])
print(f"ones:\n{ones}")
> zeros:
> [[0. 0. 0.]
>  [0. 0. 0.]]
> ones:
> [[1. 1. 1.]
>  [1. 1. 1.]]

# デフォルトの場合、行結合
np.concatenate([zeros, ones])
> array([[0., 0., 0.],
>        [0., 0., 0.],
>        [1., 1., 1.],
>        [1., 1., 1.]])

# axis=1を付けると、列結合
np.concatenate([zeros, ones], axis=1)
> array([[0., 0., 0., 1., 1., 1.],
>        [0., 0., 0., 1., 1., 1.]])
```

# 配列の最大要素のインデックスが欲しい np.argmax()
```python
a = np.arange(5)
a
> [0,1,2,3,4]
np.argmax(a)
> 4
```

