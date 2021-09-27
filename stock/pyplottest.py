import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import codecs

#csvデータの取得
def Input(NumColumns):
    datas = pd.read_csv('stock/test.csv',usecols=[NumColumns])
    return datas


def FigureOption(LabelX,LabelY,Title):
    plt.title(Title)
    plt.xlabel(LabelX)
    plt.ylabel(LabelY)
    plt.grid()

#散布図の描画
def OutFigure(X,Y,LabelX,LabelY,Title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X, Y, color='b')
    FigureOption(LabelX,LabelY,Title)
    plt.show()

x=Input(1)
y=Input(2)
print(x)
print(y)

OutFigure(x,y,"date","Price","Figure1")
