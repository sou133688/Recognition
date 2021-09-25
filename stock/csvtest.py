import numpy as np 
import matplotlib.pyplot as plt
import csv
import pandas as pd
import codecs

#csvデータの取得
def Input(NumColumns):
    datas = pd.read_csv('stock/nikkei_stock_average_daily_jp.csv', index_col=0,usecols=[NumColumns])
    # datas = pd.read_csv('stock/test.csv',usecols=[NumColumns])

    #UniCodeErrorの回避
    # with codecs.open("stock/nikkei_stock_average_daily_jp.csv", "r", "Shift-JIS", "ignore") as file:
    #     datas = pd.read_table(file, delimiter=",")
    
    return datas

def OutArray(array):
    print("------------")
    print(array)
    print(array.size)


Date=Input(0)
PriceCl=Input(1)

print("DATE")
OutArray(Date)
print("PRICE_CLOSE")
OutArray(PriceCl)

