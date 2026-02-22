import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import openpyxl

# 路径定义
image_root_dir = r"C:\Users\16780\Desktop\地基云图4天测试"
power_excel_path = r"C:\Users\16780\Desktop\resnet前一2.xlsx"
table_data_path = r"D:\WPS云盘\700759574\WPS云盘\预测3.xlsx"

# 超参数
image_size = (256, 256)
batch_size = 32
epochs = 1000
learning_rate = 1e-4
patience = 50

# 加载功率数据和表格特征数据
df_power = pd.read_excel(power_excel_path)
power_values = df_power.iloc[:, 0].values  # 第一列作为目标变量

df_table = pd.read_excel(table_data_path)
table_features = df_table.values  # 表格特征数据

# 图像转换
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])


# 自定义多输入数据集
class MultiInputDataset(Dataset):
    def __init__(self, image_dir, table_features, power_values, transform):
        self.image_paths = []
        for folder in sorted(os.listdir(image_dir)):
            folder_path = os.path.join(image_dir, folder)
            for img_file in sorted(os.listdir(folder_path)):
                self.image_paths.append(os.path.join(folder_path, img_file))

        self.table_features = table_features
        self.power_values = power_values
        self.transform = transform

        # 确保数据长度一致
        min_length = min(len(self.image_paths), len(self.table_features), len(self.power_values))
        self.image_paths = self.image_paths[:min_length]
        self.table_features = self.table_features[:min_length]
        self.power_values = self.power_values[:min_length]

    def __len__(self):
        return len(self.power_values)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)

        # 表格特征
        table_feature = torch.tensor(self.table_features[idx], dtype=torch.float32)

        # 目标值
        power = torch.tensor(self.power_values[idx], dtype=torch.float32)

        return image, table_feature, power


# 多输入回归模型（使用ResNet50）
class MultiInputResNetRegressor(nn.Module):
    def __init__(self, table_feature_dim):
        super(MultiInputResNetRegressor, self).__init__()

        # 图像分支 - ResNet50[1,2](@ref)
        self.image_backbone = models.resnet50(pretrained=True)
        # ResNet50的输出特征维度是2048[5,8](@ref)
        self.image_backbone.fc = nn.Sequential(
            nn.Linear(self.image_backbone.fc.in_features, 256),  # 调整为256维
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 表格数据分支
        self.table_branch = nn.Sequential(
            nn.Linear(table_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 融合层（调整输入维度）
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32, 128),  # 256来自图像分支，32来自表格分支
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, table_data):
        # 图像特征提取
        image_features = self.image_backbone(image)

        # 表格特征提取
        table_features = self.table_branch(table_data)

        # 特征融合
        combined_features = torch.cat([image_features, table_features], dim=1)

        # 回归输出
        output = self.fusion(combined_features).squeeze(1)
        return output


# 写入预测结果到Excel的函数
def write_predictions_to_excel(excel_path, predictions, start_row=0, column=3):
    """
    将预测结果写入Excel文件的指定列
    """
    try:
        workbook = openpyxl.load_workbook(excel_path)
        sheet = workbook.active

        for i, pred in enumerate(predictions):
            row_index = start_row + i + 1
            sheet.cell(row=row_index, column=column, value=float(pred))

        workbook.save(excel_path)
        print(f"预测结果已成功写入 {excel_path} 的第 {column} 列")

    except Exception as e:
        print(f"写入Excel文件时出错: {e}")


# 构建数据集和加载器
full_dataset = MultiInputDataset(image_root_dir, table_features, power_values, transform)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# 获取表格特征的维度
sample_image, sample_table, sample_power = full_dataset[0]
table_feature_dim = sample_table.shape[0]

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 初始化模型（使用ResNet50）
model = MultiInputResNetRegressor(table_feature_dim).cuda()
print("模型结构:")
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练参数
best_val_loss = float('inf')
early_stop_counter = 0
best_model_state = None
model_save_path = 'resnet50_multi_input_model.pth'  # 修改模型保存名称

print("开始训练多输入回归模型（使用ResNet50）...")
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for images, table_data, targets in train_loader:
        images, table_data, targets = images.cuda(), table_data.cuda(), targets.cuda()

        preds = model(images, table_data)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, table_data, targets in val_loader:
            images, table_data, targets = images.cuda(), table_data.cuda(), targets.cuda()
            preds = model(images, table_data)
            loss = criterion(preds, targets)
            val_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 早停机制和模型保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        best_model_state = model.state_dict()
        torch.save(best_model_state, model_save_path)
        print(f"最佳模型已保存为 {model_save_path}，验证损失: {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"\n早停在第 {epoch + 1} 轮。最佳验证损失: {best_val_loss:.4f}")
            break

# 加载最佳模型
if best_model_state:
    model.load_state_dict(best_model_state)
else:
    torch.save(model.state_dict(), model_save_path)
    print(f"最终模型已保存为 {model_save_path}")

# 测试集评估
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for images, table_data, targets in test_loader:
        images, table_data = images.cuda(), table_data.cuda()
        preds = model(images, table_data).cpu().numpy()
        targets = targets.numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

# 计算评估指标
mae = mean_absolute_error(all_targets, all_preds)
mse = mean_squared_error(all_targets, all_preds)
rmse = mean_squared_error(all_targets, all_preds, squared=False)
r2 = r2_score(all_targets, all_preds)

print("\n测试集评估结果:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 对所有数据进行预测并写入Excel第三列
print("\n正在生成所有数据的预测结果并写入Excel...")
model.eval()
all_predictions = []

# 创建完整数据集的DataLoader
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for images, table_data, _ in full_loader:
        images, table_data = images.cuda(), table_data.cuda()
        preds = model(images, table_data).cpu().numpy()
        all_predictions.extend(preds)

# 将预测结果写入Excel第四列
write_predictions_to_excel(power_excel_path, all_predictions, start_row=0, column=4)

print("程序执行完毕！")
print(f"模型已保存为: {model_save_path}")
print(f"预测结果已写入: {power_excel_path} 的第四列")