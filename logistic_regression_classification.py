import pandas as pd
from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("processed_dataset/2018_01_02_preprocessed.csv", header=1)

X = df.iloc[:, [0, 1, 2]].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# train the model with inbuilt classifier
def train():
    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(accuracy_score(y_test, predictions))
    dump(model, './model/lr_model_inbuilt.joblib')
    print('Model saved')


# do prediction from the saved model
def prediction(data):
    model = load('./model/lr_model_inbuilt.joblib')
    predictions = model.predict(data)

    plot_confusion_matrix(model, data, predictions)
    plt.show()

    return predictions

# train()
# prediction(X_test)
# classification_report(y_test, predictions)
# print(confusion_matrix(y_test, predictions))
