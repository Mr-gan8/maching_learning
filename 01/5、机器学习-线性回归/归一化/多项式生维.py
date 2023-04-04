#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/2 23:47
# @Author  : 蓝桉
# @File    : 多项式生维.py
# @Software: PyCharm
# pip install -i https://pypi.doubanio.com/simple/ 包名
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import  mean_squared_error

np.random.seed(42)
m = 100
