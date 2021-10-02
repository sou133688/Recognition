import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#ボストン住宅価格データを読み込み、格納したデータフレームを返す
def load_data():
    california = fetch_california_housing()

    # データフレームへ格納
    df = pd.DataFrame(california.data, columns = california.feature_names)
    df["MEDV"] = california.target
    return df

#データセットを分割し訓練データ,テストデータを返す
def learn_data():
    df = load_data()

    X = df[["MEDV"]]
    Y = df[["HouseAge"]]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    return X_train, X_test, Y_train, Y_test

# 単回帰訓練予測データを返す
def prediction_train_simple():
    X_train, X_test, Y_train, Y_test = learn_data()
    
    simple_reg = LinearRegression().fit(X_train, Y_train)
    Y_train_pred = simple_reg.predict(X_train)
    
    return Y_train_pred

# 単回帰テスト予測データを返す    
def prediction_test_simple():
    X_train, X_test, Y_train, Y_test = learn_data()
    
    simple_reg = LinearRegression().fit(X_train, Y_train)
    Y_test_pred = simple_reg.predict(X_test)
    
    return Y_test_pred

# 標準化された訓練データ,テストデータを返す
def data_scale(array):    
    scaler = StandardScaler()
    array_scaled = scaler.fit_transform(array)

    return array_scaled

# 重回帰データを返す
def multi_regression():
    X_train, X_test, Y_train, Y_test = learn_data()
    X_train_scaled = data_scale(X_train)

    multi_reg = LinearRegression().fit(X_train_scaled, Y_train)
    
    return multi_reg

# 重回帰訓練予測データを返す
def prediction_train_multi():
    X_train, X_test, Y_train, Y_test = learn_data()
    X_train_scaled = data_scale(X_train)

    multi_reg = LinearRegression().fit(X_train_scaled, Y_train)
    Y_train_pred_multi = multi_reg.predict(X_train_scaled)
    
    return Y_train_pred_multi

# 重回帰テスト予測データを返す    
def prediction_test_multi():
    X_train, X_test, Y_train, Y_test = learn_data()
    X_train_scaled = data_scale(X_train)
    X_test_scaled = data_scale(X_test)
    
    multi_reg = LinearRegression().fit(X_train_scaled, Y_train)
    Y_test_pred_multi = multi_reg.predict(X_test_scaled)
    
    return Y_test_pred_multi

def out_score(mae,mse,rmse,r2):
    print(f"MAE = {mae}")
    print(f"MSE = {mse}")
    print(f"RMSE = {rmse}")
    print(f"R2 SCORE = {r2}")

def get_eval_score(Y_true, Y_pred):
    # 平均絶対誤差
    mae = mean_absolute_error(Y_true, Y_pred)
    # 平均二乗誤差
    mse = mean_squared_error(Y_true, Y_pred)
    # 二乗平均平方根誤差
    rmse = np.sqrt(mse)
    # 決定係数
    r2 = r2_score(Y_true, Y_pred)

    out_score(mae,mse,rmse,r2)

    return 0



# X_train, X_test, Y_train, Y_test = learn_data()
# X_train_scaled = data_scale(X_train)
# X_test_scaled = data_scale(X_test)
# multi_reg = LinearRegression().fit(X_train_scaled, Y_train)





