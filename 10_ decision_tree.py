# 決定木
# オプション教材のためざっくりメモ書き程度

''' 決定木とは
yes/no で分岐していくイメージ

【決定木の学習】
選択肢nの中から最も分割できている選択肢を採用していくイメージ

【Gini係数で決定木を構築】
Gini係数で不純度を計算
Gini係数の差分が大きい方がより分割できている。
機械学習により、Gini係数の差分が最大になる物を見つけて採用していく。
'''

''' アンサンブル学習
単一の学習機を用いた学習ではどうしても限界がある。
アンサンブル学習とは複数の学習機を用いて更なる精度向上を目指す学習手法のこと。
学習機を並列、直列にして複数回学習を行う。

（ブースティング）
並列配置：ブースティング
直列配置：勾配ブースティング
直列配置＋全学習モデル使用：AdaBoost

（バギング）
並列配置：ブートストラップサンプリング

（ランダムフォレスト）
並列配置+多数決：ランダムフォレスト

（スタッキング）
並列配置を直列配置：スタッキング
'''