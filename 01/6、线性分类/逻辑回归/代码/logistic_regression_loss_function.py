from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

data = load_breast_cancer()
X, y = scale(data['data'][:, :2]), data['target']
# 求出两个维度对应的数据在逻辑回归算法下的最优解
lr = LogisticRegression(fit_intercept=False)
lr.fit(X, y)
# 分别把两个维度所对应的参数W1和W2取出来
theta1 = lr.coef_[0, 0]
theta2 = lr.coef_[0, 1]
print(theta1, theta2)


# 已知W1和W2的情况下，传进来数据的X，返回数据的y_predict
def p_theta_function(features, w1, w2):
    z = w1*features[0] + w2*features[1]
    return 1 / (1 + np.exp(-z))


# 传入一份已知数据的X，y，如果已知W1和W2的情况下，计算对应这份数据的Loss损失
def loss_function(samples_features, samples_labels, w1, w2):
    result = 0
    # 遍历数据集中的每一条样本，并且计算每条样本的损失，加到result身上得到整体的数据集损失
    for features, label in zip(samples_features, samples_labels):
        # 这是计算一条样本的y_predict
        p_result = p_theta_function(features, w1, w2)
        loss_result = -1*label*np.log(p_result)-(1-label)*np.log(1-p_result)
        result += loss_result
    return result


theta1_space = np.linspace(theta1-0.6, theta1+0.6, 50)
theta2_space = np.linspace(theta2-0.6, theta2+0.6, 50)

result1_ = np.array([loss_function(X, y, i, theta2) for i in theta1_space])
result2_ = np.array([loss_function(X, y, theta1, i) for i in theta2_space])

fig1 = plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.plot(theta1_space, result1_)

plt.subplot(2, 2, 2)
plt.plot(theta2_space, result2_)

plt.subplot(2, 2, 3)
theta1_grid, theta2_grid = np.meshgrid(theta1_space, theta2_space)
loss_grid = loss_function(X, y, theta1_grid, theta2_grid)
plt.contour(theta1_grid, theta2_grid, loss_grid)

plt.subplot(2, 2, 4)
plt.contour(theta1_grid, theta2_grid, loss_grid, 30)

fig2 = plt.figure()
ax = Axes3D(fig2)
ax.plot_surface(theta1_grid, theta2_grid, loss_grid)

plt.show()
