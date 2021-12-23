# 生成モデル

''' DCGAN (Deep convolutional GAN)
### 【GANからの変更点】
- GANに畳み込み層の概念を追加。

### 【DCGAN実装時の工夫ポイント】
- pooling layerを使用しない。
- batch normalizationを使う。
- fully-connected layerを使用しない。
- Generatorの出力の活性化関数をtanh, その他をReLUにする。
- Discriminatorの活性化関数は全てLeaky ReLUにする。
'''

''' CGAN (Conditional-GAN)
### 【GANからの変更点】
- GeneratorとDiscriminatorにonehotラベルを渡す処理が追加された。
'''