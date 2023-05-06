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
    alpha = 1e-12
    learning_rate = 1e-12
    
#     # 对数据进行归一化
#     max_vals = np.max(X, axis=0)
#     min_vals = np.min(X, axis=0)
#     X = (X - min_vals) / (max_vals - min_vals)

    # 初始化模型参数
    theta = np.zeros(6)

    # 定义L1正则化项的梯度
    def l1_grad(theta, alpha):
        return alpha * np.sign(theta)

    # 进行梯度下降迭代
    for i in range(800):
        # 计算模型预测值
        y_pred = X @ theta

        # 计算损失函数和L1正则化项
        loss = np.mean((y_pred - y)**2) + alpha * np.sum(np.abs(theta))

        # 计算梯度并更新模型参数
        grad = (np.matmul(X.T, (np.matmul(X, theta) - y))) + l1(theta,alpha)
        theta = theta - learning_rate * grad
        if np.linalg.norm(grad)<1e-5:
            break
    return data @ theta

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y





