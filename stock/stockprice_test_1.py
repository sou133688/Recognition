import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import codecs




stock_datas = pd.read_csv('stock/nikkei_stock_average_daily_jp.csv')
with codecs.open("stock/nikkei_stock_average_daily_jp.csv", "r", "Shift-JIS", "ignore") as file:
    stock_datas = pd.read_table(file, delimiter=",")

print(stock_datas.size)

# for data in stock_datas:
#     print(stock_datas[data[0],data[1]])

# for data in stock_datas:
#     plt.scatter(data[0], data[1])

# plt.title("Stock Average")
# plt.xlabel("Date")
# plt.ylabel("Price(JPY)")
# plt.grid()

for data in stock_datas:
    plt.scatter(data[0], data[1])
plt.show()
