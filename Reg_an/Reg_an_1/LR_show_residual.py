import pred_LR
import learn_dataset
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = learn_dataset.LearnData()

Y_train_pred = pred_LR.Prediction(1)
Y_test_pred = pred_LR.Prediction(0)

#訓練データの予測の可視化
plt.scatter(Y_train_pred, Y_train_pred-Y_train , label="train", color="blue")
plt.scatter(Y_test_pred, Y_test_pred-Y_test, label="test", color="green")
plt.plot([0, 50], [0, 0], color="red")
plt.xlabel("pred")
plt.ylabel("pred-true")
plt.title("Residual Plot")
plt.legend()
plt.show()