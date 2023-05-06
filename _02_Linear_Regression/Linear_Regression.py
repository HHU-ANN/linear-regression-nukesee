# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y=read_data()
    alpha=-0.1
    I = np.eye(6)
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return data @ beta
    
    
def lasso(data):
    X,y=read_data()
    alpha = 1e-12
    learning_rate = 1e-12
    theta = np.zeros(6)
    def l1(theta, alpha):
        return alpha * np.sign(theta)

    for i in range(100000):
        # 计算梯度并更新模型参数
        grad = (np.matmul(X.T, (np.matmul(X, theta) - y))) + l1(theta,alpha)
        theta = theta - learning_rate * grad
        if np.linalg.norm(grad)<1e-5:
            break
    return theta @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y





