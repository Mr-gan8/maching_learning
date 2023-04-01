import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


X1 = 2*np.random.rand(100, 1)
X2 = 2*np.random.rand(100, 1)
X = np.c_[X1, X2]

y = 4 + 3*X1 + 5*X2 + np.random.randn(100, 1)

reg = LinearRegression(fit_intercept=True)
reg.fit(X, y)
print(reg.intercept_, reg.coef_)

X_new = np.array([[0, 0],
                  [2, 1],
                  [2, 4]])
y_predict = reg.predict(X_new)

# 绘图进行展示真实的数据点和我们预测用的模型
plt.plot(X_new[:, 0], y_predict, 'r-')
plt.plot(X1, y, 'b.')
plt.axis([0, 2, 0, 25])
plt.show()
