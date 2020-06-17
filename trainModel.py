import pandas as pd
from sklearn.model_selection import train_test_split
from pybrain.datasets import SupervisedDataset

images = pd.read_csv('Dataset/images.csv')
labels = pd.read_csv('Dataset/label.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
training = SupervisedDataset(images.shape[1], y.shape[1])

