import pandas as pd
import numpy as np
import plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取Excel文件
file_path = r"D:\WPS云盘\700759574\WPS云盘\catboost集成.xlsx"

try:
    # 读取数据
    data = pd.read_excel(file_path)
    print("数据读取成功！")
    print(f"数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    print("\n数据前5行:")
    print(data.head())

except Exception as e:
    print(f"读取文件时出错: {e}")
    exit()

# 2. 数据预处理
# 假设最后一列是目标变量，前面所有列是特征
X = data.iloc[:, :-1]  # 所有行，除最后一列外的所有列
y = data.iloc[:, -1]  # 所有行，最后一列

print(f"\n特征形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 检查并处理缺失值
if data.isnull().sum().sum() > 0:
    print("\n发现缺失值，进行处理...")
    # 对数值列用中位数填充，对类别列用众数填充
    for column in X.columns:
        if X[column].dtype in ['object', 'category']:
            X[column].fillna(X[column].mode()[0] if len(X[column].mode()) > 0 else 'missing', inplace=True)
        else:
            X[column].fillna(X[column].median(), inplace=True)
    y.fillna(y.median(), inplace=True)

# 3. 识别类别特征
categorical_features = []
for i, column in enumerate(X.columns):
    if X[column].dtype in ['object', 'category']:
        categorical_features.append(i)
        print(f"检测到类别特征: {column} (索引: {i})")

if categorical_features:
    print(f"\n总共检测到 {len(categorical_features)} 个类别特征")
else:
    print("\n未检测到类别特征，所有特征均为数值型")

# 4. 划分训练集、验证集和测试集（70%-15%-15%）
# 首先将数据分为训练集（70%）和临时集（30%）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 然后将临时集平均分为验证集和测试集（各15%）
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\n训练集大小: {X_train.shape} (70%)")
print(f"验证集大小: {X_validation.shape} (15%)")
print(f"测试集大小: {X_test.shape} (15%)")

# 5. 创建并训练CatBoost模型
model = CatBoostRegressor(
    iterations=1000,  # 迭代次数
    learning_rate=0.1,  # 学习率
    depth=6,  # 树深度
    loss_function='RMSE',  # 损失函数
    eval_metric='RMSE',  # 评估指标
    random_seed=42,  # 随机种子
    verbose=100,  # 每100次迭代输出一次日志
    early_stopping_rounds=50  # 提前停止
)

print("\n开始训练模型...")
# 训练模型，使用验证集进行早停
model.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_validation, y_validation),
    plot=False,
    verbose=100,
    use_best_model=True  # 使用在验证集上表现最好的模型
)

print("模型训练完成！")

# 6. 模型评估
# 在训练集、验证集和测试集上进行预测
y_pred_train = model.predict(X_train)
y_pred_validation = model.predict(X_validation)
y_pred_test = model.predict(X_test)

# 计算评估指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
validation_rmse = np.sqrt(mean_squared_error(y_validation, y_pred_validation))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

train_r2 = r2_score(y_train, y_pred_train)
validation_r2 = r2_score(y_validation, y_pred_validation)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "=" * 60)
print("模型评估结果:")
print(f"训练集 RMSE: {train_rmse:.4f}")
print(f"验证集 RMSE: {validation_rmse:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}")
print(f"训练集 R²: {train_r2:.4f}")
print(f"验证集 R²: {validation_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")
print("=" * 60)

# 7. 对整个数据集进行预测并添加到原数据
print("\n对整个数据集进行预测...")
full_predictions = model.predict(X)

# 将预测结果添加到DataFrame中（最后一列旁边）
target_column_name = data.columns[-1]
new_column_name = f"{target_column_name}_预测"

# 创建数据的副本以避免修改原始数据
result_data = data.copy()
result_data[new_column_name] = full_predictions

# 8. 保存结果到新文件
output_file_path = file_path.replace('.xlsx', '_预测结果.xlsx')

try:
    result_data.to_excel(output_file_path, index=False)
    print(f"预测结果已保存到: {output_file_path}")

    # 显示结果的前几行
    print(f"\n结果数据前5行（包含预测列）:")
    print(result_data.head())

except Exception as e:
    print(f"保存文件时出错: {e}")
    # 如果保存失败，显示前几行结果
    print("\n预测结果预览:")
    print(result_data.head())

# 9. 特征重要性分析
print("\n特征重要性分析:")
feature_importance = model.get_feature_importance()
importance_df = pd.DataFrame({
    '特征': X.columns,
    '重要性': feature_importance
}).sort_values('重要性', ascending=False)

print(importance_df)

# 10. 学习曲线可视化
import matplotlib.pyplot as plt

# 获取训练过程中的损失值
evals_result = model.get_evals_result()

if 'learn' in evals_result and 'validation' in evals_result:
    train_errors = evals_result['learn']['RMSE']
    validation_errors = evals_result['validation']['RMSE']

    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label='训练集 RMSE')
    plt.plot(validation_errors, label='验证集 RMSE')
    plt.title('CatBoost 学习曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

# 11. 可选：特征重要性可视化
plt.figure(figsize=(10, 6))
importance_df.sort_values('重要性').plot.barh(x='特征', y='重要性')
plt.title('CatBoost 特征重要性')
plt.tight_layout()
plt.show()

print("\n程序执行完成！")