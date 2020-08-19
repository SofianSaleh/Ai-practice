import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

# test purpos

# acc = []
# for i in range(10):
df = pd.read_csv('./K_neighbors/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_jobs=1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# test purpos
# acc.append(accuracy)

example = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [
    2, 1, 5, 6, 9, 8, 5, 3, 6], [6, 5, 3, 6, 4, 10, 2, 3, 6]])
example = example.reshape(len(example), -1)
prediction = clf.predict(example)

print(accuracy, prediction)


# test purpos
# print(sum(acc)/len(acc))
# 0.9707142857142858
