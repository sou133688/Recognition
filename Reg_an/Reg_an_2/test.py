from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing

boston = load_boston()
california = fetch_california_housing()
print(boston.feature_names)
print(california.feature_names)
