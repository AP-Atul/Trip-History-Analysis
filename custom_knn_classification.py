import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lib.KNN_Classifier import KNN

# Reading the data set from the processed csv file
df = pd.read_csv("processed_dataset/2018_01_02_preprocessed.csv", header=1, low_memory=False)

X = df.iloc[0:10000, [0, 1, 2]].values
y = df.iloc[0:10000, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)


# Training the model from the data set
def train_model():
    clf = KNN()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test, k=5)
    print("Accuracy with inbuilt :: ", accuracy_score(y_test, y_pred) * 100)
    dump(clf, './model/knn_model_custom_train_k_5.joblib')
    print("Model saved.................")


# Predicting using the saved model
def predict_load_model(test_data):
    clf = load('./model/knn_model_inbuilt_k_19.joblib')
    predictions = clf.predict(test_data)
    return predictions
