# ニューラルネットワーク
"""サイキットラーンのロジスティック回帰について
【公式docs】 sklearn.linear_model.LogisticRegression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

【Qiita】 Scikit-learn でロジスティック回帰（クラス分類編）
https://qiita.com/0NE_shoT_/items/b702ab482466df6e5569

【データ科学便覧】 Scikit-learnによるロジスティック回帰
https://data-science.gr.jp/implementation/iml_sklearn_logistic_regression.html
"""
from sklearn.linear_model import LogisticRegression

def learning_curve_plot(estimator, X, y, train_sizes):
	"""# 訓練例追加によるスコアの変化をプロット

	Args:
		estimator ([type]): [description]
		X ([type]): [description]
		y ([type]): [description]
		train_sizes ([type]): [description]
	"""
	plt.figure()
	plt.title(u"学習曲線")
	plt.xlabel(u"訓練例の数")
	plt.ylabel(u"スコア")
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, n_jobs=2)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()
	plt.ylim(ymax=1)
	plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="学習スコア")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="交差検証スコア")
	plt.legend(loc="best")
	plt.show()