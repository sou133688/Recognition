import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

#指定列のcsvデータの取得
def Input(NumColumns):
    datas = pd.read_csv('stock/nikkei_stock_average_daily_jp.csv', index_col=0,usecols=[NumColumns])
    
    return datas


#日付データの取得
Date=Input(0)
print(Date)

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

print("DATE")
OutArray(Date)
print("PRICE_CLOSE")
OutArray(PriceCl)

tags_array=["DATE","PriceClose","PriceOpen"]

data_array=[]

for i in range(3):
    data_array[i]=Input(i)
    OutArray(data_array[1])