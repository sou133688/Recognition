import numpy as np 
import matplotlib.pyplot as plt
import csv
import pandas as pd
import codecs

#csvデータの取得
def Input():
    datas = pd.read_csv('stock/test.csv', header=None, nrows=3)    
    return datas


df=Input()
print(df)
