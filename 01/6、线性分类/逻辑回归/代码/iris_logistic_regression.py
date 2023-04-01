import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print(list(iris.keys()))
print(iris['DESCR'])
print(iris['feature_names'])

X = iris['data'][:, 3:]
print(X)

print(iris['target'])
# y = (iris['target'] == 2).astype(np.int)
y = iris['target']
print(y)

multi_classifier = LogisticRegression(solver='sag', max_iter=1000, multi_class='multinomial')
multi_classifier.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X_new)
y_proba = multi_classifier.predict_proba(X_new)
print(y_proba)
y_hat = multi_classifier.predict(X_new)
print(y_hat)
