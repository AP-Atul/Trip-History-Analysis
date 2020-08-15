import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("duration_cal_preprocessed.csv", header=1)
# df.head()

# df.hist()
# plt.show()

X = df.iloc[:, [0, 1, 2]].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Find Score
score = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of our model is: {:.1f}%".format(score * 100))
print(classification_report(y_test, y_pred))
