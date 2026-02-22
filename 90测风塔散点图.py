import pandas as pd
import matplotlib.pyplot as plt

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据文件路径
file_path = "C:\\Users\\16780\\Desktop\\tffs.xlsx"

# 加载Excel数据到DataFrame
df = pd.read_excel(file_path)

# 提取所需的列
times = df.iloc[:, 0]  # 第一列时间
station_windspeed = df.iloc[:, 1]  # 气象站风速
tower_windspeed = df.iloc[:, 2]  # 测风塔风速

# 绘制散点图
plt.figure(figsize=(10, 6))  # 设置图形大小

# 气象站风速散点图
plt.scatter(times, station_windspeed, label="气象站风速", color="blue")

# 测风塔风速散点图
plt.scatter(times, tower_windspeed, label="测风塔风速", color="red")

# 添加图表元素
plt.xlabel("时间")
plt.ylabel("风速 (m/s)")
plt.legend()  # 显示图例

# 设置横坐标间隔
plt.xticks(rotation=45)  # 旋转横坐标标签，使其斜着显示
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # 设置横坐标主刻度数量

# 显示图形
plt.show()
