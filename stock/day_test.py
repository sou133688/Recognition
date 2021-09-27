import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

#指定行のcsvデータの取得
def Input():
    datas = pd.read_csv('stock/date.csv', header=None, index_col=0, usecols=[0])
    
    return datas


#日付データの取得
x=Input()
print(x)
print(x.dtyeps)

# 横軸：日付 periods分の日付を用意します。


# 縦軸：数値
y = [130, 141, 142, 143, 171, 230, 231, 260, 276, 297]
print(y)
print(y.dtypes)

ax.plot(x,y)

# 日付ラベルフォーマットを修正
days = mdates.DayLocator() 
daysFmt = mdates.DateFormatter('%m-%d')
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(daysFmt)

# グラフの表示
plt.show()