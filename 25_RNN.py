#RNN

''' 式の展開
h(t):時刻tにおける系の隠れ状態
θ   :計算グラフのパラメータ

h(t) = f(h(t-1);θ)

【1ステップ後】
h(t+1) = f(h(t);θ)

【入力x(t)を導入する】
h(t) = f(h(t-1),x(t);θ)

h(t)   ← 新しい隠れ状態
f      ← θをパラメータとしたある関数f
h(t-1) ← 1ステップ前の隠れ状態
'''

''' RNNの順伝播
x(t)    : 時刻ｔにおける入力
W, U, V : 重み
b, c    : バイアス
f, g    : 活性化関数

fはtanhなど、gはソフトマックスなどを使う。

a(t) = b + Wh(t-1) + Ux(t)
h(t) = f(a(t))
o(t) = c+Vh(t)
y_hat(t) = g(o(t))
'''

''' BPTT（通時的誤差逆伝播法）
全然わからん

計算式はともかく、
RNNは1つのレイヤーのW,bはシェアしてるところが特徴なのだとか。

RNN入門PART2：レイヤ・誤差計算・BPTT
https://www.youtube.com/watch?v=DWectS03wg8
'''

''' BPTT以外の誤差逆電波法
（BPTTの問題点：長い時系列データを扱うと勾配消失、勾配爆発を起こす。）

- Truncated BPTT
- 教師強制と出力回帰のあるネットワーク
'''

''' 様々なRNNのモデル
- 有向グラフィカルモデル結合RNN
- 双方向RNN
    手書き文字認識、音声認識、生物情報学等に応用されている。
- Encoder-Decoder
- Attention Model
- ESN(Echo State Network)
'''

''' 長期依存性の処理
- ゲート付きRNN
 - LSTM（Long short-term memory）
    https://www.youtube.com/watch?v=oxygME2UBFc&list=PLhDAH9aTfnxKXf__soUoAEOrbLAOnVHCP&index=11
 - GRU（Gated recurrent unit）
    要約：「前回の情報をどれだけ忘れるか」を制御して長期記憶を持たせる。
    https://www.youtube.com/watch?v=K8ktkhAEuLM&list=PLhDAH9aTfnxKXf__soUoAEOrbLAOnVHCP&index=10
'''
