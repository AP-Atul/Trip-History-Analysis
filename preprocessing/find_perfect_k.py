import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

df = pd.read_csv("processed_dataset/2018_01_02_preprocessed.csv", header=1, low_memory=False, )

X = df.iloc[:100000, [0, 1, 2]].values
y = df.iloc[:100000, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Find a perfect K
K = []
training = []
test = []
scores = {}

for k in tqdm(range(2, 21)):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    training_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    K.append(k)

    training.append(training_score)
    test.append(test_score)
    scores[k] = [training_score - test_score]

# Evaluating the model
for key, values in scores.items():
    print(key, " : ", values)

ax = sns.stripplot(K, training)
ax.set(xlabel='values of k', ylabel='Training Score')
plt.show()

ax = sns.stripplot(K, test)
ax.set(xlabel='values of k', ylabel='Test Score')
plt.show()

plt.scatter(K, training, color='k')
plt.scatter(K, test, color='g')
plt.show()
