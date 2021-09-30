import learn_dataset
from sklearn.linear_model import LinearRegression

def Prediction(flag):
    X_train, X_test, Y_train, Y_test = learn_dataset.LearnData()
    simple_reg = LinearRegression().fit(X_train, Y_train)
    
    Y_test_pred = simple_reg.predict(X_test)
    Y_train_pred = simple_reg.predict(X_train)

    if (flag==1):
        return Y_train_pred
    
    else:
        return Y_test_pred
        
        
Y_train_pred = Prediction(1)
Y_test_pred = Prediction(2)

print(len(Y_train_pred))
print(Y_train_pred[:5])
print(len(Y_test_pred))
print(Y_test_pred[:5])

