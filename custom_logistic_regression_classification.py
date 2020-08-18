import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lib.Logistic_Regression_Classifier import LR

df = pd.read_csv("processed_dataset/2018_01_02_preprocessed.csv", header=1)

X = df.iloc[:, [0, 1, 2]].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# train the model with inbuilt classifier
def train():
    model = LR()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(accuracy_score(y_test, predictions))
    dump(model, 'model/lr_model_custom.joblib')
    print('Model saved')


# do prediction from the saved model
def prediction(data):
    model = load('model/lr_model_custom.joblib')
    predictions = model.predict(data)
    return predictions


def controller_predict(controller, test_data, test_labels):
    clf = load('model/lr_model_custom.joblib')
    predictions = clf.predict(test_data)
    controller.setLRCustom(round(accuracy_score(test_labels, predictions) * 100, 3))

# train()
# y_pred = prediction(X_test)
# print(accuracy_score(y_test, y_pred))
# classification_report(y_test, predictions)
# print(confusion_matrix(y_test, predictions))
