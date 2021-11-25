""" カーネル法
知らないカーネルが出てきたら調べると良いらしい
「 svm scikit-learn 」 で検索がおすすめ

scikit-learn svm docs
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""

""" どのモデルを使用したらよいかの判断をどうするか
特徴の数nが多い時
ロジスティック回帰、カーネル無しSVM
nが少なく、訓練例の数mが普通の時
ガウシアンカーネルSVM
nが少なく、mが多い時
多項式にして、ロジスティック回帰、カーネル無しSVM
"""
