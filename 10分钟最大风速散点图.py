import pandas as pd
import matplotlib.pyplot as plt

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据文件路径
file_path = "C:\\Users\\16780\\Desktop\\10zuidafs.xlsx"

# 加载Excel数据到DataFrame
df = pd.read_excel(file_path)

# 创建原始数据副本
df_original = df.copy()

# 提取所需的列
times = df.iloc[:, 0]  # 第一列时间
station_windspeed = df.iloc[:, 1]  # 气象站风速
tower_windspeed = df.iloc[:, 2]  # 测风塔风速

# 绘制散点图
plt.figure(figsize=(10, 10))  # 设置图形大小

# 气象站风速散点图
plt.scatter(times, station_windspeed, label="测风塔90m60分钟平均风速", color="blue")

# 测风塔风速散点图
plt.scatter(times, tower_windspeed, label="测风塔90m10分钟最大风速", color="red")

# 添加图表元素
plt.xlabel("时间")
plt.ylabel("风速 (m/s)")
plt.legend()  # 显示图例


# 显示图形
plt.show()

