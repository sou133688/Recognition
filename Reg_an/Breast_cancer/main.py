import package_datasets as pcd
import package_algorithm as pca
import numpy as np

# model = input("Enter model name : ")

df = pcd.load_data()

X_train, X_test, Y_train, Y_test = pcd.learn_data(df, "mean radius", "mean texture", "y")
X_train_scaled = pcd.data_scale(X_train)
X_test_scaled = pcd.data_scale(X_test)

Y_test=np.reshape(Y_test,(-1,1))
Y_train=np.reshape(Y_train,(-1,1))

print(Y_test)
print(Y_train)

# -----------------------------------------------------------------------------------------

Y_train_pred = pca.logistic_regression_model(X_train_scaled, Y_train, X_train_scaled, "pred")
Y_test_pred = pca.logistic_regression_model(X_train_scaled, Y_train, X_test_scaled, "pred")
log_reg = pca.logistic_regression_model(X_train_scaled, Y_train, X_test_scaled, "model")

# shp=Y_train.shape
# print(shp)

print(Y_train_pred[:5])
print(Y_test_pred[:5])


pcd.plot_decision_regions(np.array(X_train_scaled), np.array(Y_train), clf=log_reg)
pcd.plt.show()

# pcd.plot_data(X_test["mean radius"], X_test["mean texture"], Y_test_pred, "mean radius", "mean texture", "Pred Test")
# pcd.plot_data(X_train["mean radius"], X_train["mean texture"], Y_train_pred, "mean radius", "mean texture", "Pred Train")

# print(X_test_scaled[:3])

# print(len(X_train))
# display(X_train.head())
# print(len(X_test))
# display(X_test.head())
