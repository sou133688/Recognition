import pred_LR
import learn_dataset
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = learn_dataset.LearnData()

Y_train_pred = pred_LR.Prediction(1)
Y_test_pred = pred_LR.Prediction(0)

#訓練データ,テストデータの予測の可視化
plt.scatter(X_train, Y_train_pred, label="train")
plt.scatter(X_test, Y_test_pred, label="test")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Simple Liner Regression")
plt.legend()
plt.show()
