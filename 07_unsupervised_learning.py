# 教師なし学習
''' k-means法（k平均法）
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

手順
共分散行列を計算する
Σ = 1/m Σ:n:i=1 (x(i))(x(i))T

共分散行列Σの固有値を計算
U, S, V = np.linalg.svd(Sigma)
'''

''' よく分らんやつ
- 特異値分解（SVD）の数式
- 半正定値エルミーと行列
- カルバックライブラー情報量
'''

''' 主成分分析の実装
Sigma = X.T@X / m
U, S, V = np.linalg.svd(Sigma)
Ureduce = U[:, :k]
z = Ureduce.T@X
'''

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
