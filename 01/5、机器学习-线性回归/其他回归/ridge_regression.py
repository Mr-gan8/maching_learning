import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

X = 2*np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)

ridge_reg = Ridge(alpha=0.4, solver='sag')
ridge_reg.fit(X, y)
# print(ridge_reg.predict([[1.5]]))
# print(ridge_reg.intercept_)
# print(ridge_reg.coef_)

# sgd_reg = SGDRegressor(penalty='l2', max_iter=10000)
# sgd_reg.fit(X, y.reshape(-1,))
# print(sgd_reg.predict([[1.5]]))
# print(sgd_reg.intercept_)
# print(sgd_reg.coef_)
