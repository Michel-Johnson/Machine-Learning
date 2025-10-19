import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 搭配博客解析，阅读更简单 http://micheljohnson.top/2025/09/16/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/

#******************************************************************************************************
#单变量
path =  'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 绘制原始数据散点图
ax = data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
# 对捕获到的 Axes 对象设置标签
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Scatter plot of single-variable original data') # 也可以设置标题

# 显示图表
plt.show()


#--------------------------------------------------------------------------------------------
#创建一个以参数θ为特征函数的代价函数
def computeCost(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2)
    return (np.sum(inner)) / (2 * len(X))
#--------------------------------------------------------------------------------------------

data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]

X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]#y是所有行，最后一列
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))


print("单变量学习前：",computeCost(X, y, theta))


#--------------------------------------------------------------------------------------------
#Batch gradient descent
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X @ theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost
#--------------------------------------------------------------------------------------------

# change alpha and iters
alpha = 0.01
iters = 1000

g, cost = gradientDescent(X, y, theta, alpha, iters)
print("单变量学习后：",computeCost(X, y,g))


x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)


# 查看拟合
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Univariate fitting scatter plot')
plt.show()


#查看cost变化
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


#******************************************************************************************************
# 多变量
path =  'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std() #Z-score normalization!!!
#print(data2.head()) #查看数据
#--------------------------------------------------------------------------------------------
# --- 查看数据，绘制三维散点图 ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d') # 创建一个三维坐标轴
ax.scatter(data2['Size'], data2['Bedrooms'], data2['Price'], c='blue', marker='o', label='House data')
# 设置轴标签
ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
ax.set_title('Three-dimensional scatter plot of housing price data')
ax.legend()
plt.show()
#--------------------------------------------------------------------------------------------
# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

print("多变量学习前：",computeCost(X2, y2, theta2))

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
print("多变量学习后：",computeCost(X2, y2, g2))



# --- 生成三维拟合图像 --- (代码由AI生成）
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d') # 创建一个三维坐标轴

ax.scatter(np.array(data2['Size']).flatten(),
           np.array(data2['Bedrooms']).flatten(),
           np.array(data2['Price']).flatten(),
           c='blue', marker='o', label='House data')

x_surf = np.linspace(data2['Size'].min(), data2['Size'].max(), 50)
y_surf = np.linspace(data2['Bedrooms'].min(), data2['Bedrooms'].max(), 50)
X_surf, Y_surf = np.meshgrid(x_surf, y_surf)

# 计算平面上的 Z 值 (预测价格)
# 模型的方程是 Price = theta0 + theta1 * Size + theta2 * Bedrooms
# g2 是 (1, 3) 矩阵 [theta0, theta1, theta2]
Z_surf = g2[0, 0] + g2[0, 1] * X_surf + g2[0, 2] * Y_surf

# 绘制拟合平面
# cmap='viridis' 设置颜色映射，alpha=0.5 设置透明度
ax.plot_surface(X_surf, Y_surf, Z_surf, cmap='viridis', alpha=0.6, label='Fitted Plane')

# 设置轴标签
ax.set_xlabel('Size (Standardized)')
ax.set_ylabel('Bedrooms (Standardized)')
ax.set_zlabel('Price (Standardized)')
ax.set_title('Three-dimensional scatter plot with Fitted Plane')

# 查看cost变化---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

#******************************************************************************************************
#Normal Equaltion
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X)
    return theta

final_theta2=normalEqn(X, y)
print ("final theta--Normal Equaltion:",final_theta2)

print("单变量--Normal Equaltion：",computeCost(X, y, final_theta2.T))
#******************************************************************************************************