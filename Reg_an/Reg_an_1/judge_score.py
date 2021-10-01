# テスト,訓練データの評価指標

import pred_LR
import learn_dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train, X_test, Y_train, Y_test = learn_dataset.LearnData()
Y_train_pred = pred_LR.Prediction(1)
Y_test_pred = pred_LR.Prediction(0)

def get_eval_score(Y_test, Y_test_pred):
    # 平均絶対誤差
    mae = mean_absolute_error(Y_test, Y_test_pred)
    # 平均二乗誤差
    mse = mean_squared_error(Y_test, Y_test_pred)
    # 二乗平均平方根誤差
    rmse = np.sqrt(mse)
    # 決定係数
    r2 = r2_score(Y_test, Y_test_pred)

    return mae,mse,rmse,r2


def out_score(mae,mse,rmse,r2):
    print(f"MAE = {mae}")
    print(f"MSE = {mse}")
    print(f"RMSE = {rmse}")
    print(f"R2 SCORE = {r2}")

mae_te,mse_te,rmse_te,r2_te = get_eval_score(Y_test, Y_test_pred)
print("Test Data Score")
out_score(mae_te,mse_te,rmse_te,r2_te)

mae_tr,mse_tr,rmse_tr,r2_tr = get_eval_score(Y_train, Y_train_pred)
print("Train Data Score")
out_score(mae_tr,mse_tr,rmse_tr,r2_tr,0)
