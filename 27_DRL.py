# 深層強化学習

''' 概要
比較的教師なし学習に近い手法。
何かしらの行った結果とスコアを基に、
最大スコアが獲得できるように行動するように学習する。
'''

# ↓強化学習のアルゴリズム

''' モデルベースとモデルフリー違い
モデルベース
- （遷移行列と報酬関数）をベースに学習
- モデルは既知である、あるいは推定する必要がある。

モデルフリー
- モデルは不要
- エージェントの経験を基に学習する。
'''

''' 動的計画法
複雑な問題を解くためのアルゴリズム。  
次の２つの性質がある場合、この手法で解くことが出来る。  
1. 問題を部分問題に分解し、それらを解くことで元の問題を解くことが出来る。
   (分割統治法:Divide-and-Conquer Model)
2. 部分問題の解放が他の部分問題で再利用可能である。
   (メモ化:Memorization)

マルコフ決定過程も動的計画法採用に必要な2要件を満たしている。  

動的計画法はモデルベースの学習方法  
具体的には以下２つのアルゴリズム  
1. 価値反復法
2. 方策反復法
'''

''' 価値反復法(Value Iteration Methods)
★価値が最大となる行動を選ぶ
実際に状態価値関数を見つける手法。  
※ 状態、行動、報酬が有限で、遷移行列が分かっている必要がある。  
'''

''' 方策反復法(Policy Iteration Methods)
★エージェントの方策に基づき行動を選ぶ
Vに関するベルマン方程式を使う。
'''

''' モンテカルロ法
モデルフリー系
★エピソードが修了してから方策を改善
方策πに従って、時刻t=0からt=Tまでのエピソードを作成

【実装】
input:方策π
init:
    V(s)∈R, ∀s∈S
    Returns(s)←空のリスト, ∀s∈S
for エピソード分
    方策πに従ってエピソードをk作成: S0,A0,R1,S1,A1,...,ST-1,AT-1,RT
    G←0
    for エピソード中のステップ分, t=T-1, T-2,...,0
        G←γG + Rt+1
        if 時刻t以前に同じ状態が無ければ(=初回訪問)
            Returns(S1)にGを追加
            V(St)←average(Returns(St))
'''

''' TD法
モデルフリー系
★時間が進むごとに方策を改善出来る
'''

''' Sarsa
TD法系
'''

''' Q学習
TD法系
'''

''' DQN
Deep Network と Q学習 を組み合わせたもの
'''

''' より新しい手法
Rainbow
A2C
DDPG
TRPO
'''
