# 教師なし学習
''' k-means法（k平均法）

k-means法解説
【機械学習】クラスタリングとは何か(k-means)
<https://www.youtube.com/watch?v=8yptHd0JDlw>

重心を更新してクラスタリングする方法
（重心を更新しても移動しなくなったら収束）

【用意するもの】
入力: クラスタの数 K
訓練例：{x1, x2, xm} ∈ Rn
クラスタ重心（K個）： μ1, μ2, ... , μK ∈ Rn

【計算過程】
以下繰り返し
for i=1 to m
    c(i):= 訓練例x(i)に最も近いクラスタ重心のインデックス(1-K)
for k=1 to K
    μk:= クラスタkに割り当てられた訓練例の平均

【目的関数】
J(c(1), ..., c(m), μ1, ..., μK) = 1/m Σ|| x(i) - μc(i) ||^2

【学習過程】
min J(c(1~m), μ1~K)
c(1~m)
μ1~K

【k-means法の局所最適解問題】
重心をどこからスタートするかで最適解が変化する。→凸性が保証されていない
対策
- 何通りものランダム初期化計算をする（50~1,000あたりが妥当）
手順
1. ランダム初期化
2. k-means法のアルゴリズムを走らせてc(1~m), μ1~Kを得る
3. 目的関数を計算する
4. 最後に目的関数が最低のクラスタリングを採用する
※ クラスタ数Kが2~10個ほど（クラスタ数Kが少ないときに）有効な方法

【k-means++】
最初からよりよい初期値をセットしてくれるk-means法

【クラスタ数はどのように決定したらよい？】
- エルボー法
クラスタ数を増やしていき、SSE値の減少率が飽和し始める点=最適なクラスタ数、とする手法
- シルエット分析
データの密集度合いのグラフからクラスタ数を決定する方法
'''

''' DBSCAN
ある点を中心とした円を考え、そこから到達可能な点群は同一クラスタとみなす手法
密度準拠型といわれる
クラスタ数は自動的に決まる

【ハイパーパラメータ】
1. ε: コア点からの距離
2. Minimum points: コアがグループを構成する最小点数
'''

''' 主成分分析
【次元削減】
目的
1. PCのメモリ使用量をセーブする
2. 学習アルゴリズムの高速化
3. データの可視化

【PCA】
Principal Components Analysis

【次元圧縮】主成分解析（PCA）の理論
<https://www.youtube.com/watch?v=mTmuwLF45T8>

手順
共分散行列を計算する
Σ = 1/m Σ:n:i=1 (x(i))(x(i))T

共分散行列Σの固有値を計算
U, S, V = np.linalg.svd(Sigma)
'''

''' よく分らんやつ
- 特異値分解（SVD）の数式
- 半正定値エルミーと行列  
'''

''' 主成分分析の実装
Sigma = X.T@X / m
U, S, V = np.linalg.svd(Sigma)
Ureduce = U[:, :k]
z = Ureduce.T@X
'''

''' カルバック・ライブラー情報量
SNEの損失関数として利用されている
<https://ai999.careers/rabbit/eshikaku/4-1-機械学習における損失関数とは？/>

【SNE損失関数の計算式】
カルバック・ライブラー情報量の和
C = Σi KL(P||Q) = ΣiΣj≠i pj|i log pj|i/qj|i
'''

''' t-SNE
【次元圧縮】t-SNE (t -distributed Stochastic Neighborhood Embedding)の理論
<https://www.youtube.com/watch?v=hEanlhXGoME>

【特徴】
非線形で局所構造を保つのに優れている。
（PCAは線形で対局構造を保つのに優れている）
【批判】
Preplexityが異なると異なるクラスターを形成することがある。
まだまだ改善の余地が残っている。
'''

#####################################
# k-means法の重心プロットテンプレート #
#####################################
import matplotlib as plt

def plotData(X, idx=np.ones(X.shape[0],dtype=float)):
    map = plt.get_cmap("rainbow")
    idxn = idx.astype("float") / max(idx.astype("float"))
    colors = map(idxn)
    plt.scatter(X[:, 0], X[:, 1], 15, marker="o", c=colors, edgecolors=colors)
    plt.draw()
def plotProgress(X, centroids, idx=np.ones(X.shape[0],dtype=float)):
    plt.hold(True)
    plotData(X, idx)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x",s=200,linewidths=4, c="g")
    plt.show()

#################
# k-means法実装 #
#################
import numpy as np

def InitCentroids(X, K):
    """ 訓練集合からランダムなK個のデータを取得

    Args:
        X (np.ndarray): 訓練集合
        K (int): 取得したいデータの個数

    Returns:
        np.ndarray: ランダムなK個のデータ
    """
    m, _ = X.shape
    idx = np.random.permutation(np.arange(m))
    centroids = X[idx[:K],:]
    return centroids

def findClosestCentroids(X, centroids):
    """ 各訓練例に対して最も近い重心を返す  

    Args:
        X (np.ndarray): 訓練集合
        centroids (np.ndarray): K個のクラスター重心

    Returns:
        np.ndarray: 各訓練例に対して最も近い重心
    """
    K,_ = centroids.shape
    m, _ = X.shape
    idx = np.empty(m, dtype=int)
    for i in range(m):
        norm = np.empty(K)
        for j in range(K):
            norm[j] = np.linalg.norm(X[i,:] - centroids[j,:])
        idx[i] = np.argmin(norm)
    return idx

def computeClusterMeans(X, idx):
    """各クラスターのデータ平均を求める

    Args:
        X (np.ndarray): 訓練集合
        idx (np.ndarray)): 訓練集合が属するクラスター集合

    Returns:
        np.ndarray: 各クラスターのデータ平均
    """
    K = idx.max()+1
    clusterMeans = np.array([X[idx==i,:].mean(axis=0) for i in range(K)])
    return clusterMeans[~np.isnan(clusterMeans).any(axis=1)] #nanを含む行は除外して返す

''' k-means法の学習
centroids = InitCentroids(X, 3)
idx = findClosestCentroids(X, centroids)
plotProgress(X,centroids,idx)
for i in range(1,6):
    centroids = computeClusterMeans(X, idx)
    idx = findClosestCentroids(X, centroids)
    plotProgress(X,centroids,idx)
'''

def runKmeans(X, K_init, epsilon, max_iter):
    """ k-means法を繰り返す関数

    Args:
        X (np.ndarray): 訓練集合
        K_init (int): 取得したいデータの個数
        epsilon (float): ?
        max_iter (int): 学習回数

    Returns:
        centroids (np.ndarray): 各クラスターの重心
        idx (np.ndarray): 各データの属するクラスター番号
    """
    centroids_previous = InitCentroids(X, K_init)
    idx = findClosestCentroids(X, centroids_previous)
    centroids = computeClusterMeans(X, idx)
    idx = findClosestCentroids(X, centroids)
    num_iter = 0
    while np.linalg.norm(centroids - centroids_previous, axis=1).mean() > epsilon and num_iter < max_iter:
        centroids_previous = num_iter
        centroids = computeClusterMeans(X, idx)
        idx = findClosestCentroids(X, centroids)
        num_iter += 1
    return (centroids, idx)

###########################
# scikit-learnでk-means法 #
###########################
import sklearn.cluster as skc

# インスタンス生成
estimator = skc.KMeans(n_clusters=3, n_jobs=-1) #n_jobs=-1でCPUの全コア利用。デフォルトでは1コア利用。
# 学習(k-means法)
estimator.fit(X)
# 学習結果をプロットしてみる
idx = estimator.predict(X)
centroids = estimator.cluster_centers_
plotProgress(X,centroids,idx)

#########################
## 画像の色素数を落とす ##
#########################
A = scm.imread("{画像のパス}").astype(np.float32)/255
#intで0から255で表現するのではなく、float64で0から1で表現することで、k平均法フィッティング時の型関係のトラブルを回避。
#しかしメモリ節約にはならない。気になる人はもっとよい方法を考えよう。

plt.axis("off")
plt.imshow(A)
img_size = A.shape
print(img_size)
Aflatten = A.reshape(img_size[0]*img_size[1], 3) # 2次元データに変換

# クラスタリング
K = 5 # 色素数
estimator_pic = skc.KMeans(n_clusters=K, n_jobs=-1)
estimator_pic.fit(Aflatten)
color_idx = estimator_pic.predict(Aflatten)
color_centroids = estimator_pic.cluster_centers_
# 復元
Aflatten_recovered = np.array([color_centroids[i] for i in color_idx])
A_recovered = Aflatten_recovered.reshape(img_size)
plt.axis("off")
plt.imshow(A_recovered)

###########
# PCA実装 #
###########
def featureScaling(X):
    """ 各特徴のレンジを揃える

    Args:
        X (np.ndarray): 訓練集合

    Returns:
        X_norm (np.ndarray): ノルム
        mu (np.ndarray)    : 平均
        std (np.ndarray)   : 標準偏差
    """
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mu) / std
    return [X_norm, mu, std]

def runPCA(X):
    """ PCA
    np.linagle.svdについて
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html  
    https://qiita.com/kyoro1/items/4df11e933e737703d549

    Args:
        X (np.ndarray): 訓練集合

    Returns:
        U (np.ndarray): m*m のユニタリ行列
        S (np.ndarray): m*n の実対角行列
    """
    m, _ = X.shape
    Sigma = X.T@X / m
    U, S, _ = np.linalg.svd(Sigma)
    return [U, S]
