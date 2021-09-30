from sklearn.datasets import load_boston
import pandas as pd
from IPython.display import display

#ボストン住宅価格データを読み込み、格納したデータフレームを返す
def load_data():
    boston = load_boston()

    print("説明変数")
    print(f"{len(boston.data)}件")
    print(boston.data[:5])

    print("目的変数")
    print(f"{len(boston.target)}件")
    print(boston.target[:5])

    print("変数名")
    print(f"{len(boston.feature_names)}件")
    print(boston.feature_names[:5])

    # データフレームへ格納
    df = pd.DataFrame(boston.data, columns = boston.feature_names)
    df["MEDV"] = boston.target
    return df

df=load_data()    
display(df.head())