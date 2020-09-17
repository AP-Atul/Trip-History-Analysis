import pandas as pd
from joblib import load, dump
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# """Data set reading"""
df = pd.read_csv("processed_dataset/2018_01_02_preprocessed.csv", header=1, low_memory=False, )

X = df.iloc[:, [0, 1, 2]].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)


# Training model using sk learn classifier
def train_model():
    clf = KNeighborsClassifier(n_neighbors=19, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy with inbuilt :: ", accuracy_score(y_test, y_pred) * 100)
    dump(clf, 'model/knn_model_inbuilt_k_19.joblib')
    print("Model saved.................")


# Predicting using the saved model
def predict_load_model(X_test, plot=True):
    clf = load('model/knn_model_inbuilt_k_19.joblib')
    predictions = clf.predict(X_test)

    if plot:
        plot_confusion_matrix(clf, X_test, predictions)
        plt.show()
    return predictions


def controller_predict(controller, test_data, test_labels, plot=True):
    clf = load('model/knn_model_inbuilt_k_19.joblib')
    predictions = clf.predict(test_data)
    
    if plot:
        plot_confusion_matrix(clf, test_data, predictions)
        plt.show()

    controller.setKNNInbuilt(round(accuracy_score(test_labels, predictions) * 100, 3))

# train_model()
# ypred = predict_load_model(X_test, plot=False)
# print(accuracy_score(y_test, ypred))
