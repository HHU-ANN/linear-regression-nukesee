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
    alpha = 0.1
    learning_rate = 0.001

    # 初始化模型参数
    theta = np.zeros(6)

    # 定义L1正则化项的梯度
    def l1_grad(theta, alpha):
        return alpha * np.sign(theta)

    # 进行梯度下降迭代
    for i in range(10):
        # 计算模型预测值
        y_pred = X @ theta

        # 计算损失函数和L1正则化项
        loss = np.mean((y_pred - y)**2) + alpha * np.sum(np.abs(theta))

        # 计算梯度并更新模型参数
        grad = (X.T @ (y_pred - y)) / len(y) + l1_grad(theta, alpha)
        theta = theta - learning_rate * grad
    return data @ theta

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y





