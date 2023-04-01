import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

X = 2*np.random.randn(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# elastic_reg = ElasticNet(alpha=0.04, l1_ratio=0.15)
# elastic_reg.fit(X, y)
# print(elastic_reg.predict([[1.5]]))

sgd_reg = SGDRegressor(penalty='elasticnet', max_iter=1000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))
