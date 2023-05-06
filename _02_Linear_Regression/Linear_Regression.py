# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y=read_data()
    alpha=0.11
    I = np.eye(6)
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return data @ beta
    
    
def lasso(data):
    
    X,y=read_data()
    m, n = X.shape
    weight = np.zeros(n)
    max_iterations = 100000
    for i in range(max_iterations):
        grad = (np.matmul(X.T, (np.matmul(X, weight) - y))) + 1e-12 * np.sign(weight)
        weight = weight - 1e-12 * grad
        if np.linalg.norm(grad) < 0.0001:
            break
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y





