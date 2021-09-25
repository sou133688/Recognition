import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import codecs

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

x=[1,3,5]
y=[2,4,6]
OutFigure(x,y,"date","Price","Figure1")
