import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("duration_cal_preprocessed.csv", header=1)
# df.head()

# df.hist()
# plt.show()

X = df.iloc[:, [0, 1, 2]].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
classification_report(y_test, predictions)
print(confusion_matrix(y_test, predictions))
