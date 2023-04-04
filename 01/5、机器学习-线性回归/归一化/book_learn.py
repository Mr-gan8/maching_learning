#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 15:03
# @Author  : 蓝桉
# @File    : book_learn.py
# @Software: PyCharm
# pip install -i https://pypi.doubanio.com/simple/ 包名
from sklearn.preprocessing import StandardScaler
import numpy as np
temp = np.array([1,2,3,5,5])
temp = temp.reshape(-1,1)
print(temp)

scaler = StandardScaler()
print(scaler.fit(temp))
print(scaler.mean_)
print(scaler.var_)
print(scaler.scale_)
print(scaler.transform(temp))
data = np.array([1,2,3,5,50001]).reshape(-1,1)
print(scaler.fit(data))
print(scaler.var_)
print(scaler.mean_)
print(scaler.transform(data))
