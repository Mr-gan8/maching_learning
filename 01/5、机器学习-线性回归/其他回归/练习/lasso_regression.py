#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 20:31
# @Author  : 蓝桉
# @File    : lasso_regression.py
# @Software: PyCharm
# pip install -i https://pypi.doubanio.com/simple/ 包名
import numpy as np
from sklearn.linear_model import  Lasso
from sklearn.linear_model import SGDRegressor
X = 2*np.random.randn(100,1)
y = 4 +3 * X + np.random.randn(100,1)
# lasso_reg = Lasso(alpha=0.15, max_iter=30000)
# lasso_reg.fit(X,y)
# print(lasso_reg.predict([[1.5]]))
# print(lasso_reg.intercept_)
# print(lasso_reg.coef_)
sgd_reg = SGDRegressor(penalty='l1',max_iter=10000)
sgd_reg.fit(X,y)
print(sgd_reg.predict([[1.5]]))
print(sgd_reg.intercept_)
print(sgd_reg.coef_)