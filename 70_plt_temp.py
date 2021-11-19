import numpy as np
import matplotlib as plt

def plotData(X, y):
    """ # サンプル（部屋の広さと賃料の関係）

    基本的には下の２行でグラフを表示することができますが、
    ちょっとした装飾テクニックを知っておくと便利です。
    plt.scatter(X,y)
    plt.draw()

    Args:
        X (numpy.ndarray): 行列
        y (numpy.ndarray): 列ベクトル
    """
    plt.xlim([0,25]) #x軸の範囲を指定
    plt.ylim([0,35]) #y軸の範囲を指定 
    plt.xlabel('部屋の広さ[畳]',fontsize=14) #x軸のラベル
    plt.ylabel('家賃 [万円]', fontsize=14) #y軸のラベル
    plt.title('部屋の広さと賃料の関係', fontsize=16) #グラフのタイトル
    plt.scatter(X, y, c='b') #散布図
    plt.draw() #グラフを表示

def out_minmax_plot(x):
    if np.amin(x) < 0:
        x_min = np.amin(x) * 1.1
    else:
        x_min = np.amin(x) * 0.5
    if np.amax(x) < 0:
        x_max = np.amax(x) * 0.5
    else:
        x_max = np.amax(x) * 1.1
    return x_min, x_max

def tow_axes_plot_temp(x, y, title=None, xlabel=None, ylabel=None):

    x_min, x_max = out_minmax_plot(x)
    y_min, y_max = out_minmax_plot(y)

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.scatter(x, y)
    plt.draw()