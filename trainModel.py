import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pickle

images = pd.read_csv('Dataset/images.csv')
labels = pd.read_csv('Dataset/label.csv')

clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=14)

X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=0.9)
clf.fit(X_train, y_train)

filename = 'MultiPerceptronModel.sav'
pickle.dump(clf, open(filename, 'wb'))

def print_clf():
    print(len(clf.coefs_))
    print(clf.coefs_[0].shape)
    print(clf.coefs_[1].shape)


y_pred = clf.predict(X_test)


def print_result():
    f1score = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    classificationreport = classification_report(y_pred=y_pred, y_true=y_test)
    print("f1 Score: \n")
    print(f1score)
    print("Classification Report: \n")
    print(classificationreport)


print_result()



