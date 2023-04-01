import numpy as np


# 创建数据集X，y
np.random.seed(1)
X = np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# 创建超参数
n_iterations = 10000

t0, t1 = 5, 500


# 定义一个函数来调整学习率
def learning_rate_schedule(t):
    return t0/(t+t1)


# 1,初始化θ, W0...Wn，标准正太分布创建W
theta = np.random.randn(2, 1)

# 4,判断是否收敛，一般不会去设定阈值，而是直接采用设置相对大的迭代次数保证可以收敛
for i in range(n_iterations):
    # 2,求梯度，计算gradient
    gradients = X_b.T.dot(X_b.dot(theta)-y)
    # 3,应用梯度下降法的公式去调整θ值 θt+1=θt-η*gradient
    learning_rate = learning_rate_schedule(i)
    theta = theta - learning_rate * gradients

print(theta)
