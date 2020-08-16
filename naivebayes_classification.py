import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("processed_dataset/2018_01_02_preprocessed.csv", header=1)
# df.head()

# df.hist()
# plt.show()

X = df.iloc[:, [0, 1, 2]].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB

nv = GaussianNB()  # create a classifier
nv.fit(X_train, y_train)

y_pred = nv.predict(X_test)  # store the prediction data
print("Naive Bayesr score :: ", accuracy_score(y_test, y_pred) * 100)
#
# df = pd.DataFrame(y_pred)
# df.to_csv("prediction_nb.csv", index=True)
# print(accuracy_score(y_test, y_pred) * 100)

# nb = summarize_by_class(X_train)
# y_pred = predict(nb, X_test)
#
# print("Custom NB score :", accuracy_score(y_test, y_pred) * 100)
