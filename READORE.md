# ライブラリ
## scikit-learn
- 線形回帰を含めたその他様々な機械学習が出来るライブラリ
```python
from sklearn.linear_model import LinearRegression

# インスタンス生成
estimator = LinearRegression()

# 短回帰の学習
estimator.fit(X, y)

# min=(-3) max=3 で回帰係数の両端を出力
line_X = np.array([[-3],[3]])
line_y = estimator.predict(line_X)
```