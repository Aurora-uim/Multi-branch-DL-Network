import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from data84.pre_1d import load_data_1d
# from data84.pre_2d import load_data_2d
from data84.pre_2d import create_3d_maps
# from data28.load_1d import load_data_1d
from Model.CS_MVANet import CS_MVANet
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# load data
X_eeg, labels = load_data_1d(fs=128, window_size=4, step_size=2) # (2436, 512, 16)
# X_eeg, labels = load_data_1d(data_path='./data28/DATA.mat', seed=42) # (10052, 512, 19)
X_eeg = torch.tensor(X_eeg, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

loaded = np.load('./data84/3d_maps.npz')
spatial_tem_maps = loaded['spatial_temporal']
spatial_freq_maps = loaded['spatial_frequency']
new_labels = loaded['labels']

X_time = np.expand_dims(spatial_tem_maps.reshape(-1, spatial_tem_maps.shape[2], spatial_tem_maps.shape[3], spatial_tem_maps.shape[4]), axis=-1)# (2436, 64, 32, 32, 1)
X_freq = np.expand_dims(spatial_freq_maps.reshape(-1, spatial_freq_maps.shape[2], spatial_freq_maps.shape[3], spatial_freq_maps.shape[4]), axis=-1)

# 6:2:2
X_eeg_train, X_eeg_temp, labels_train, labels_temp = train_test_split(X_eeg, labels, test_size=0.4, random_state=42)
X_eeg_val, X_eeg_test, labels_val, labels_test = train_test_split(X_eeg_temp, labels_temp, test_size=0.5, random_state=42)
print(f"X_eeg_train shape: {X_eeg_train.shape}, labels_train shape: {labels_train.shape}")
print(f"X_eeg_val shape: {X_eeg_val.shape}, labels_val shape: {labels_val.shape}")
print(f"X_eeg_test shape: {X_eeg_test.shape}, labels_test shape: {labels_test.shape}")

X_time_train, X_time_temp, _, _ = train_test_split(X_time, labels, test_size=0.4, random_state=42)
X_time_val, X_time_test, _, _ = train_test_split(X_time_temp, labels_temp, test_size=0.5, random_state=42)
print(f"X_time_train shape: {X_time_train.shape}, labels_train shape: {labels_train.shape}")
print(f"X_time_val shape: {X_time_val.shape}, labels_val shape: {labels_val.shape}")
print(f"X_time_test shape: {X_time_test.shape}, labels_test shape: {labels_test.shape}")

X_freq_train, X_freq_temp, _, _ = train_test_split(X_freq, labels, test_size=0.4, random_state=42)
X_freq_val, X_freq_test, _, _ = train_test_split(X_freq_temp, labels_temp, test_size=0.5, random_state=42)

batch_size = 16
train_dataset_eeg = TensorDataset(torch.tensor(X_eeg_train, dtype=torch.float32), torch.tensor(labels_train, dtype=torch.long))
val_dataset_eeg = TensorDataset(torch.tensor(X_eeg_val, dtype=torch.float32), torch.tensor(labels_val, dtype=torch.long))
test_dataset_eeg = TensorDataset(torch.tensor(X_eeg_test, dtype=torch.float32), torch.tensor(labels_test, dtype=torch.long))

train_loader_eeg = DataLoader(train_dataset_eeg, batch_size=batch_size, shuffle=True)
val_loader_eeg = DataLoader(val_dataset_eeg, batch_size=batch_size, shuffle=False)
test_loader_eeg = DataLoader(test_dataset_eeg, batch_size=batch_size, shuffle=False)

train_dataset = TensorDataset(
    torch.tensor(X_time_train, dtype=torch.float32),
    torch.tensor(X_freq_train, dtype=torch.float32),
    torch.tensor(labels_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_time_val, dtype=torch.float32),
    torch.tensor(X_freq_val, dtype=torch.float32),
    torch.tensor(labels_val, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_time_test, dtype=torch.float32),
    torch.tensor(X_freq_test, dtype=torch.float32),
    torch.tensor(labels_test, dtype=torch.long)
)
train_loader_3D = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader_3D = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader_3D = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CS_MVANet(eeg_channels=16, in_channels=1, emb_size=32, num_heads=4, L=64, W=32, H=32, num_classes=2).to(device)
# model = CS_MVANet(eeg_channels=19, in_channels=1, emb_size=32, num_heads=4, L=64, W=32, H=32, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True, min_lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for (X_eeg_batch, y_batch), (X_time_batch, X_freq_batch, _) in zip(train_loader_eeg, train_loader_3D):  # 加载EEG和3D
        X_eeg_batch, y_batch = X_eeg_batch.to(device), y_batch.to(device)
        X_time_batch, X_freq_batch = X_time_batch.to(device), X_freq_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_eeg_batch, X_time_batch, X_freq_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total

    # val
    model.eval()
    val_loss = 0
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for (X_eeg_batch, y_batch), (X_time_batch, X_freq_batch, _) in zip(val_loader_eeg, val_loader_3D):
            X_eeg_batch, y_batch = X_eeg_batch.to(device), y_batch.to(device)
            X_time_batch, X_freq_batch = X_time_batch.to(device), X_freq_batch.to(device)

            outputs = model(X_eeg_batch, X_time_batch, X_freq_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)

    val_acc = val_correct / val_total

    current_lr = optimizer.param_groups[0]['lr']
    print("\n" +
          f"Epoch [{epoch + 1}/{num_epochs}] | "
          f"Train Loss: {total_loss / len(train_loader_3D):.6f}, Train Acc: {train_acc * 100:.2f}% | "
          f"Val Loss: {val_loss / len(val_loader_3D):.6f}, Val Acc: {val_acc * 100:.2f}% | "
          f"Lr: {current_lr:.4f}")

    scheduler.step(val_loss)

# test
model.eval()
all_preds ,all_labels=[],[]
test_loss = 0
test_correct, test_total = 0, 0
with torch.no_grad():
    for (X_eeg_batch, y_batch), (X_time_batch, X_freq_batch, _) in zip(test_loader_eeg, test_loader_3D):
        X_eeg_batch, y_batch = X_eeg_batch.to(device), y_batch.to(device)
        X_time_batch, X_freq_batch = X_time_batch.to(device), X_freq_batch.to(device)

        outputs = model(X_eeg_batch, X_time_batch, X_freq_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == y_batch).sum().item()
        test_total += y_batch.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

test_acc = test_correct / test_total
print(f"Test Loss: {test_loss / len(test_loader_3D):.4f}, Test Acc: {test_acc * 100:.2f}%")

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
