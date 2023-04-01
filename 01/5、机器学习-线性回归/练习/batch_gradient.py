#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 9:58
# @Author  : 蓝桉
# @File    : batch_gradient.py
# @Software: PyCharm
# pip install -i https://pypi.doubanio.com/simple/ 包名
import numpy as np
theta = np.random.randn(2,1)
X_1 = np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)),X_1]
y =  3+7*X_1+np.random.randn(100,1)
iterations = 10000
t0,t1 = 5, 500
for i in range(iterations):
    gradients = X_b.T.dot(X_b.dot(theta)-y)
    learning_rate = t0/(t0+i)
    theta = theta - learning_rate*gradients
print(theta)