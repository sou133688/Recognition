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

#csvデータの取得
def Input():
    datas = pd.read_csv('stock/nikkei_stock_average_daily_jp.csv')
    
    #UniCodeError回避
    with codecs.open("stock/nikkei_stock_average_daily_jp.csv", "r", "Shift-JIS", "ignore") as file:
        datas = pd.read_table(file, delimiter=",")
    
    return datas



stock_table=Input();

print(stock_table)
print(stock_table.size)

PriceOp=2
PriceCl=1
PriceHi=3
PriceLo=4



# for data in stock_datas:
#     print(stock_datas[data[0],data[1]])

# for data in stock_datas:
#     plt.scatter(data[0], data[1])


# for data in stock_datas:
#     plt.scatter(data[0], data[1],data[2])
# plt.show()
