import dataset
from IPython.display import display

# display(X.head())
# display(Y.head())

from sklearn.model_selection import train_test_split

#訓練データ,テストデータを返す
def LearnData():
    df = dataset.load_data()

    X = df[["RM"]]
    Y = df[["MEDV"]]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = LearnData()

print(len(X_train))
display(X_train.head())
print(len(X_test))
display(X_test.head())