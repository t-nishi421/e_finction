# functools docs
<https://docs.python.org/ja/3/library/functools.html>

# functools.partial(部分適用)について
- Pythonのfunctools.partial(部分適用)を解説する
<https://qiita.com/tetsuya-zama/items/146d3ee9154e15158f38>

- 【Python】functools.partial()で関数やメソッドの引数の一部を固定する部分適用を行う
<https://www.st-hakky-blog.com/entry/2018/02/01/225734>

【解説】
```python
def func(A, B, C):
    return A + B * C

'''
みたいな関数があったとして、
本当にfunc(A=A, B=B, C=C)って形で使われる保証ってある？
func(C, A, B)って形で呼ぶこともできるよね？

functools.partial()を使うと
そんないちゃもんをつけられることは無い。
'''

# 【使い方】
# functools.partial(関数, 引数1, 引数2, ..., 引数n)
from functools import partial

A=10
B=3
C=4

# こうすることで、A, B, Cが固定されたfunc関数が出来る
partial_obj = partial(func, A=A, B=B, C=C)

print(partial_obj())
# > 22
```
