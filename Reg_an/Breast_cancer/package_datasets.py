from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import package_algorithm as pca


#乳癌診断データを読み込み,"mean radius","mean texture"を格納したデータフレームを返す
def load_data():
    load_data = load_breast_cancer()
    df = pd.DataFrame(load_data.data, columns=load_data.feature_names)
    df["y"]=load_data.target
    
    tg_df = df[["mean radius","mean texture","y"]]

    return tg_df 

def plot_data(data_1, data_2, data_3, X_label, Y_label, Title):
    plt.scatter(data_1, data_2, c=data_3)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(Title)
    plt.colorbar()
    plt.show()

#データセットを分割し訓練データ,テストデータを返す
def learn_data(df,item_1,item_2,item_3):
    X = df[[item_1,item_2]]
    Y = df[[item_3]]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    return X_train, X_test, Y_train, Y_test

# 標準化された訓練データ,テストデータを返す
def data_scale(array):    
    
    scaler = StandardScaler()
    array_scaled = scaler.fit_transform(array)

    return array_scaled

def plot(x_array, y_array, model):
    plot_decision_regions(np.array(x_array), np.array(y_array), clf=model)
    plt.show()

