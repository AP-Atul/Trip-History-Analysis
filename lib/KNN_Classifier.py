from collections import Counter

from scipy.spatial import distance
from tqdm import tqdm


class KNN:
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test, k=5):
        predictions = []
        for row in tqdm(X_test):
            label = self.closest(row, k)
            predictions.append(label)
        return predictions

    def closest(self, row, k):
        distances = []
        for i in range(len(self.X_train)):
            distances.append((i, distance.euclidean(row, self.X_train[i])))
        distances = sorted(distances, key=lambda x: x[1])[0:k]
        k_indices = []
        for i in range(k):
            k_indices.append(distances[i][0])
        k_labels = []
        for i in range(k):
            k_labels.append(self.Y_train[k_indices[i]])
        c = Counter(k_labels)
        return c.most_common()[0][0]
