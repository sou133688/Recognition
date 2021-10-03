from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ロジスティック回帰モデルの構築, 訓練orテストデータに対し予測値を返す
def logistic_regression_model(x_train_scaled, y_train, x_scaled,flag):
    log_reg = LogisticRegression(random_state=0).fit(x_train_scaled, y_train)
    y_pred = log_reg.predict(x_scaled)

    if (flag=="pred"): return y_pred
    else: return log_reg

# 線形SVMモデルの構築
def liner_svm_model(x_train_scaled, y_train, x_scaled,flag):
    lin_svm = LinearSVC(random_state=0).fit(x_train_scaled, y_train)
    y_pred = lin_svm.predict(x_scaled)

    if (flag=="pred"): return y_pred
    else: return lin_svm

# カーネルSVMモデルの構築
def kernel_svm_model(x_train_scaled, y_train, x_scaled,flag):
    kernel_svm = SVC(kernel="rbf",random_state=0).fit(x_train_scaled, y_train)
    y_pred = kernel_svm.predict(x_scaled)

    if (flag=="pred"): return y_pred
    else: return kernel_svm

# K近傍法モデルの構築
def kn_cls_model(x_train_scaled, y_train, x_scaled,flag):
    kn_cls = KNeighborsClassifier(n_neighbors=5, p=2).fit(x_train_scaled, y_train)
    y_pred = kn_cls.predict(x_scaled)

    if (flag=="pred"): return y_pred
    else: return kn_cls

# 決定木モデルの構築
def tree_cls_model(x_train, y_train, x_scaled,flag):
    tree_cls = DecisionTreeClassifier(max_depth=3, random_state=0).fit(x_train, y_train)
    y_pred = tree_cls.predict(x_scaled)

    if (flag=="pred"): return y_pred
    else: return tree_cls

# ランダムフォレストモデルの構築
def rf_cls_model(x_train_scaled, y_train, x_scaled,flag):
    rf_cls = RandomForestClassifier(max_depth=3, random_state=0).fit(x_train_scaled, y_train)
    y_pred = rf_cls.predict(x_scaled)

    if (flag=="pred"): return y_pred
    else: return rf_cls