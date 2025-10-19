import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D # 导入三维绘图工具
from matplotlib.patches import Patch # 用于自定义图例

# ******************************************************************************************************
# 多变量数据加载和标准化
path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# 保存原始的均值和标准差，以便后续反标准化（如果需要）
original_means = data2.mean()
original_stds = data2.std()

data2 = (data2 - original_means) / original_stds # Z-score normalization!!!
print("标准化后数据头：\n", data2.head())

# 1. 准备数据
# X 包含 Size 和 Bedrooms 特征，y 是 Price
X_multi = data2[['Size', 'Bedrooms']].values # 提取特征列并转换为 NumPy 数组
y_multi = data2['Price'].values               # 提取目标列

# --- 多项式曲线拟合 (多变量) ---

# 定义多项式阶数
degree = 2 # 例如，2次多项式

# 2. 创建多项式特征生成器
# include_bias=False 避免生成额外的常数项（因为 LinearRegression 默认会添加截距）
poly_multi = PolynomialFeatures(degree=degree, include_bias=False)

# 3. 将原始特征转换为多项式特征
X_multi_poly = poly_multi.fit_transform(X_multi)

# 打印转换后的特征示例 (可选)
print(f"\n原始多变量特征 X_multi.shape: {X_multi.shape}") # (n_samples, 2)
print(f"多项式多变量特征 X_multi_poly.shape: {X_multi_poly.shape}") # (n_samples, 5) for degree=2
# 对于 degree=2, 2个原始特征 (Size, Bedrooms)，会生成 5 个多项式特征:
# Size, Bedrooms, Size^2, Bedrooms^2, Size*Bedrooms
print(f"原始多变量特征前5行:\n{X_multi[:5]}")
print(f"多项式多变量特征前5行:\n{X_multi_poly[:5]}")
print(f"多项式特征名称: {poly_multi.get_feature_names_out(['Size', 'Bedrooms'])}")


# 4. 创建并训练线性回归模型 (使用多项式特征)
model_multi_poly = LinearRegression()
model_multi_poly.fit(X_multi_poly, y_multi)

# 5. 获取模型参数
print(f"\n多项式回归 (degree={degree}) 模型参数 (多变量):")
print(f"截距 (theta0): {model_multi_poly.intercept_:.4f}")
print(f"系数 (对应 {poly_multi.get_feature_names_out(['Size', 'Bedrooms'])}): {model_multi_poly.coef_}")

# 6. 评估模型
y_train_pred_multi_poly = model_multi_poly.predict(X_multi_poly)
mse_multi_poly = mean_squared_error(y_multi, y_train_pred_multi_poly)
print(f"多项式回归 (degree={degree}, 多变量) 在训练数据上的均方误差 (MSE): {mse_multi_poly:.4f}")


# 7. 绘制三维拟合曲面
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据散点图
ax.scatter(data2['Size'], data2['Bedrooms'], data2['Price'],
           c='blue', marker='o', label='House data', alpha=0.7)

# 创建网格点用于绘制曲面
# x_surf 对应 Size，y_surf 对应 Bedrooms
x_surf = np.linspace(data2['Size'].min(), data2['Size'].max(), 50)
y_surf = np.linspace(data2['Bedrooms'].min(), data2['Bedrooms'].max(), 50)
X_surf, Y_surf = np.meshgrid(x_surf, y_surf)

# 将网格点数据转换为适合 predict 的格式
# 注意：这里需要将 X_surf 和 Y_surf 展平，然后堆叠成 (n_points, 2) 的二维数组
# 再进行多项式特征转换
xy_grid = np.c_[X_surf.ravel(), Y_surf.ravel()] # 将网格点组合成 (n_points, 2)
xy_grid_poly = poly_multi.transform(xy_grid)    # 转换为多项式特征

# 使用训练好的模型进行预测
Z_surf = model_multi_poly.predict(xy_grid_poly)
Z_surf = Z_surf.reshape(X_surf.shape) # 将预测结果重新整形为 (50, 50) 以匹配网格

# 绘制拟合曲面
ax.plot_surface(X_surf, Y_surf, Z_surf, cmap='viridis', alpha=0.6, label='Fitted Surface')

# 设置轴标签
ax.set_xlabel('Size (Standardized)')
ax.set_ylabel('Bedrooms (Standardized)')
ax.set_zlabel('Price (Standardized)')
ax.set_title(f'Multi-variable Polynomial Regression (Degree {degree}) Fit')

# 自定义图例
custom_legend = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='House data'),
    Patch(facecolor=plt.cm.viridis(0.6), edgecolor='k', label=f'Fitted Surface (Degree {degree})', alpha=0.6)
]
ax.legend(handles=custom_legend, loc='upper left')

plt.show()