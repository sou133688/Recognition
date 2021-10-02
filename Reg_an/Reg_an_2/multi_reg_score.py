# 重回帰モデルの評価

import package_dataset as pcd
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = pcd.learn_data()
y_train_pred = pcd.prediction_train_multi()
y_test_pred = pcd.prediction_test_multi()

print("--MULTI EVALUATE--")
print("Train Data Score")
pcd.get_eval_score(y_train, y_train_pred)
print("Test Data Score")
pcd.get_eval_score(y_test, y_test_pred)

y_train_pred_si = pcd.prediction_train_simple()
y_test_pred_si = pcd.prediction_test_simple()

print("--SIMPLE EVALUATE--")
print("Train Data Score")
pcd.get_eval_score(y_train, y_train_pred_si)
print("Test Data Score")
pcd.get_eval_score(y_test, y_test_pred_si)
