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

