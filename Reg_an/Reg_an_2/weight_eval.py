import package_dataset as pcd

X_train, X_test, Y_train, Y_test = pcd.learn_data()
multi_reg = pcd.multi_regression()
    
print(multi_reg.coef_[0])
print(multi_reg.coef_[1])

for i,(col,coef) in enumerate(zip(pcd.load_data().columns, multi_reg.coef_[0])):
    print(f"w{i}({col})={coef}")
