# scipyの関数：expit
<http://arduinopid.web.fc2.com/Q5-3.html>

```Python
##############################
#シグモイド関数の実装に関する解説
#
#scipy.special.expit(x)は、シグモイド関数1/(1+e^(-x))の計算を行う関数です。
#xの部分にXwithBiasとthetaの行列演算を挿入しましょう。
#行列演算にはdot()関数を用います。
#気を付けてほしいのが、配列の形です。XwithBias.shapeは(100, 3)、initial_theta.shape(3,)です。行列演算が成立するように順番を考えましょう。

import scipy.optimize as scopt

scopt.minimize("{目的関数}", "{仮説}", jac="{目的関数の勾配}", args=(X,y), method='BFGS', options={'maxiter': 400, 'disp': True})
##############################
```

# scipy.optimize.minimizeの解説
科学技術計算モジュール SciPy 解説 (1) 最適化
<https://www.youtube.com/watch?v=QafNNoYjRSU>
