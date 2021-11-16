# ロジスティック回帰
import numpy as np

"""[summary]
【目的関数】
J = (-1/m) * ( y.@np.log(h) + (1-y)@np.log(1-h) )

【勾配】
grad = 1/m * (h-y)@X
"""

