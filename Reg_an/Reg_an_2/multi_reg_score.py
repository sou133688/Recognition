# 重回帰モデルの評価

from Reg_an.Reg_an_2.package_dataset import learn_data
import package_dataset as pcd
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = learn_data()
y_train_pred = pcd.prediction_train_multi()
y_test_pred = pcd.prediction_test_multi()

print("Train Data Score")
pcd.get_eval_score(y_train, y_train_pred)
print("Test Data Score")
pcd.get_eval_score(y_test, y_test_pred)
