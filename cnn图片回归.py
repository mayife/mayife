import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 路径定义
image_root_dir = r"C:\Users\16780\Desktop\地基云图4天测试"
power_excel_path = r"C:\Users\16780\WPSDrive\700759574\WPS云盘\TRANSTEST1.xlsx"

# 超参数
image_size = (256, 256)
batch_size = 8
epochs = 500
learning_rate = 1e-4

# 加载功率数据
df = pd.read_excel(power_excel_path)
power_values = df['并网功率'].values

# 图像转换
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

# 自定义数据集
class ImagePowerDataset(Dataset):
    def __init__(self, image_dir, power_values, transform):
        self.image_paths = []
        for folder in sorted(os.listdir(image_dir)):
            folder_path = os.path.join(image_dir, folder)
            for img_file in sorted(os.listdir(folder_path)):
                self.image_paths.append(os.path.join(folder_path, img_file))
        self.power_values = power_values
        self.transform = transform

    def __len__(self):
        return min(len(self.image_paths), len(self.power_values))

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        power = torch.tensor(self.power_values[idx], dtype=torch.float32)
        return image, power

# CNN-based回归模型
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)

# 构建数据集和加载器
full_dataset = ImagePowerDataset(image_root_dir, power_values, transform)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 模型训练
model = CNNRegressor().cuda()
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, targets in train_loader:
        images, targets = images.cuda(), targets.cuda()
        preds = model(images)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.cuda(), targets.cuda()
            preds = model(images)
            loss = criterion(preds, targets)
            val_loss += loss.item() * images.size(0)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss / train_size:.4f}, Val Loss: {val_loss / val_size:.4f}")

# 测试集评估
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.cuda()
        preds = model(images).cpu().numpy()
        targets = targets.numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

mae = mean_absolute_error(all_targets, all_preds)
mse = mean_squared_error(all_targets, all_preds)
rmse = mean_squared_error(all_targets, all_preds, squared=False)
r2 = r2_score(all_targets, all_preds)

print("\nTest Set Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")
