# 応用数学
### 行列計算  
- 固有値の計算
- 固有ベクトルの計算
【固有値、固有ベクトルの求め方】  
https://www.geisya.or.jp/~mwm48961/linear_algebra/eigenvalue2.htm  

### ベルヌーイ分布
2値の確率変数（例：表 or 裏） 
```
Φ∈[0,1]  

P(x=1) = Φ  
P(x=0) = 1-Φ  
```
tips:マルチヌーイ分布は多項分布のこと（例：サイコロの出目1~6）  

### ガウス分布(正規分布)
正規分布の基礎的な知識まとめ  
https://ai-trend.jp/basic-study/normal-distribution/normal-distribution/  

### ベイズの定理
例のやつ

### 特異値分解（SDV:Singular Value Deconposition）
良く分らん^^;  
SVD（特異値分解）解説  
https://qiita.com/sakami/items/d01fa353b4e1f48623a8  

詳しくはこれを買うと分かる（らしい）  
https://www.amazon.co.jp/ゼロから作るDeep-Learning-―自然言語処理編-斎藤-康毅/dp/4873118360  

### ノルム
ベクトル空間に対して「距離」を与えるための数学の道具のようなもの。  
【機械学習】LPノルムってなんだっけ？  
https://qiita.com/kenmatsu4/items/cecb466437da33df2870  

### カルバックライブラー情報量（KL-divergence）
良く分らん^^;  
https://www.youtube.com/watch?v=BHMTOffcvuE  
Kullback-Leibler Divergenceについてまとめる  
https://yul.hatenablog.com/entry/2019/01/07/152738  

### フロベニウスノルム
全成分の二乗和のルートのこと  
```
例
[[-2, 3, 2]
 [-1, 0, 4]
 [ 1,-1, 0]] → 6
```

### 情報量の計算
確率Pと置くと、log2(P)で計算できる。
エントロピー（entropy）入門 ～情報量とエントロピー～  
https://wwws.kobe-c.ac.jp/deguchi/sc180/info/ent0.html  

### 分布（用語だけ）
- 周辺確率
- 条件付き確率
- 同時確率分布
- 一様分布
- 混合分布
- 独立分布

### シャノンエントロピー I(x)
I(x) = -logP(x)

### シャノンエントロピー誤差
H(x) = -Σi P(xi)log(P(xi))

# 機械学習
### ロジスティック回帰
ロジスティック回帰の目的関数  
https://tokkan.net/ml/costfunction.html  

### SVM
- cost関数はヒンジ関数
- 目的関数のCを大きくすると外れ値の影響を小さくできる（低バイアス、高バリアンスになる）
- 線形分離不可能なケースはカーネル法を用いる(3軸グラフで閾値分類をするイメージ)

### PReLU (変数付きReLU)
https://atmarkit.itmedia.co.jp/ait/articles/2005/20/news010.html

### Adam
Deep Learning精度向上テクニック：様々な最適化手法 #1  
https://www.youtube.com/watch?v=q933reMpvX8  

# CNN
### フーリエ変換
https://www.yukisako.xyz/entry/fourier-transform

### R-CNN, Fast R-CNN, Faster R-CNN
シンプルなCNNだと無駄が多かったので検出領域を特定して効率化  
https://jp.mathworks.com/videos/object-detection-and-recognition-using-faster-r-cnn-1494263953252.html  
物体検出についての歴史まとめ(1)  
https://qiita.com/mshinoda88/items/9770ee671ea27f2c81a9  

### SegNet
SegNet: 画像セグメンテーションニューラルネットワーク  
https://qiita.com/cyberailab/items/d11862852eccc17585e8  

# RNN

# 生成モデル

# 強化学習

# プレミニ
## 勾配降下法最適化手法
### AdaGrad
過去の勾配の2乗和を蓄積していき、その平方根で学習率を割ることで学習率を調整する手法。  
問題点：学習を繰り返すうちに学習率が小さくなりすぎてしまう。
### RMSprop
移動指数平均を使うことによって過去の古い勾配の情報は無視して、直近の勾配の情報だけ見ることが出来るようになり、AdaGradの問題は解決した。
### Adadelta
パラメータの更新に使う値の次元が合わない問題を解消した。
### Adam
人気があるoptimizerでモメンタムとAdagradを組み合わせたような手法。
### Eve
Adamを改良した手法。  
### ニュートン法
2次のテイラー級数展開を用いる手法。

# プレミニ2
## DenseNet, DenseBlock
DenseNetの論文を読んで自分で実装してみる  
https://qiita.com/koshian2/items/01bd9f08444799625607  

## Batch Normalization, Layer Normalization
Layer Normalizationを理解する  
https://data-analytics.fun/2020/07/16/understanding-layer-normalization/  

### Instance Normalization
Batch Normalizationとその派生の整理  
https://gangango.com/2019/06/16/post-573/  

### MobileNet
画像認識タスクにおける計算効率の高いネットワーク  
スマホなどの小型端末でも使用できる。 
【MobileNet(v1,v2,v3)を簡単に解説してみた】
https://qiita.com/omiita/items/77dadd5a7b16a104df83   
  
Depthwise Separable ConvolutionではDepthwise ConvolutionとPointwise Convolutionそれぞれで演算を行う。  

### pix2pix
【Pix2Pix：CGANによる画像変換】  
https://blog.negativemind.com/2019/12/29/pix2pix-image-to-image-translation-with-conditional-adversarial-networks/ 
  
### U-net
共通の特徴量をレイヤー間スキップにより共有できる。  

### DenseNet, ResNet
代表的モデル「ResNet」、「DenseNet」を詳細解説！  
https://deepsquare.jp/2020/04/resnet-densenet/  

### ConditionalGAN(条件付きGAN)
画像生成時にパラメータを付与して生成する画像を指定できる。いわゆるone-hotベクトル付与型のGAN  

### プルーニング
精度を出来るだけ抑えながら過剰な重みを減らしていくプロセス。  
https://japan.xilinx.com/html_docs/vitis_ai/1_2/pruning.html  

### WaveNet
https://qiita.com/MasaEguchi/items/cd5f7e9735a120f27e2a  

# E資格出題内容の参考






