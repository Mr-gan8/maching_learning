#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 19:09
# @Author  : 蓝桉
# @File    : ridge_regression.py
# @Software: PyCharm
# pip install -i https://pypi.doubanio.com/simple/ 包名
from sklearn.linear_model import  Ridge
import numpy as np
from sklearn.linear_model import  SGDRegressor
X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)

# ridge_reg = Ridge(alpha=0.4,solver='sag')
# # sag 随机梯度下降法
# ridge_reg.fit(X,y)
#
# print(ridge_reg.predict([[1.5]])) # 预测值
# print(ridge_reg.intercept_) # 常系数也就是截距项
# print(ridge_reg.coef_) # 系数即斜率

sgd_reg = SGDRegressor(penalty= 'l2',max_iter=10000)
sgd_reg.fit(X,y.reshape(-1,))
print(sgd_reg.predict([[1.5]]))
print(sgd_reg.intercept_)
print(sgd_reg.coef_)