import numpy as np

X = 2*np.random.rand(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

n_epochs = 10000
m = 100
t0, t1 = 5, 500


# 定义一个函数来调整学习率
def learning_rate_schedule(t):
    return t0/(t+t1)


theta = np.random.randn(2, 1)
for epoch in range(n_epochs):
    # 在双层for循环之间，每个轮次开始分批次迭代之前打乱数据索引顺序
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]
    for i in range(m):
        xi = X_b[i:i+1]
        yi = y[i:i+1]
        gradients = xi.T.dot(xi.dot(theta)-yi)
        learning_rate = learning_rate_schedule(epoch*m + i)
        theta = theta - learning_rate * gradients

print(theta)
