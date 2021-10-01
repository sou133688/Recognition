#ボストンデータ価格のロード

from sklearn.datasets import load_boston
import pandas as pd
from IPython.display import display

#ボストン住宅価格データを読み込み、格納したデータフレームを返す
def load_data():
    boston = load_boston()

    # データフレームへ格納
    df = pd.DataFrame(boston.data, columns = boston.feature_names)
    df["MEDV"] = boston.target
    return df

def Show(df):
    display(df.head())
    return 0

df=load_data()    
Show(df)