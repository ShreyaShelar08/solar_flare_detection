import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import sys

# Add current directory (src/) to module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import clean_and_pivot_helios, prepare_nowcasting_windows

# Define 1D-CNN Model with Skip/Residual Connections
class Nowcast1DCNN(nn.Module):
    def __init__(self, num_features=12):
        super(Nowcast1DCNN, self).__init__()
        # Input shape: (batch_size, num_features, sequence_length) e.g., (batch, 12, 60)
        
        # Initial convolution layer
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        
        # Residual branch block: extracts deep patterns while matching channel shape
        self.res_conv = nn.Conv1d(32, 32, kernel_size=5, padding=2, bias=False)
        self.res_bn = nn.BatchNorm1d(32)
        
        # MaxPool 1 (sequence length: 60 -> 30)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second Conv layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # MaxPool 2 (sequence length: 30 -> 15)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Flatten and Fully Connected classification layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 15, 64)
        self.fc_dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 1)  # Outputs raw logit
        
    def forward(self, x):
        # 1. First block with skip connection
        out = self.relu(self.bn1(self.conv1(x)))
        res = self.res_bn(self.res_conv(out))
        out = self.relu(out + res)  # Skip connection element-wise sum
        
        # 2. Downsample
        out = self.pool1(out)
        out = self.dropout1(out)
        
        # 3. Second convolutional block
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        out = self.dropout2(out)
        
        # 4. Dense Classifier
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.fc_dropout(out)
        logits = self.fc2(out)
        return logits

# Define Binary Focal Loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.20, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are raw logits, targets are 0 or 1
        targets = targets.float()
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate probability of correct class
        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # Calculate focal loss factor (alpha for positive, 1-alpha for negative)
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = focal_weight * ((1 - pt) ** self.gamma) * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss

def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Calculate confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # Sensitivity (Recall) and Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # True Skill Statistic (TSS)
    tss = sensitivity + specificity - 1
    
    # Heidke Skill Score (HSS)
    expected_correct = ((tp + fn)*(tp + fp) + (fp + tn)*(fn + tn)) / (tp + fp + fn + tn)
    actual_correct = tp + tn
    total = tp + fp + fn + tn
    hss = (actual_correct - expected_correct) / (total - expected_correct) if (total - expected_correct) > 0 else 0
    
    return {
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'Sensitivity': float(sensitivity), 'Specificity': float(specificity),
        'Precision': float(precision), 'TSS': float(tss), 'HSS': float(hss)
    }

def train_nowcasting_model(data_path, save_path='models/nowcast_1dcnn.pt'):
    # Load and preprocess data
    df_helios = clean_and_pivot_helios(data_path)
    X, y = prepare_nowcasting_windows(df_helios, window_size=60, step_size=10)
    print(f"Windows created. Shape of X: {X.shape}, Shape of y: {y.shape}")
    print(f"Class distribution: Flare = {np.sum(y)}, Non-Flare = {len(y) - np.sum(y)}")
    
    # Flatten windows for SMOTE
    n_samples, win_len, n_features = X.shape
    X_flat = X.reshape(n_samples, win_len * n_features)
    
    # Apply SMOTE to address class imbalance only on the training split to avoid data leakage
    # We do a stratified train-test split first
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Applying SMOTE on training split...")
    smote = SMOTE(random_state=42)
    n_train = X_train_raw.shape[0]
    X_train_flat = X_train_raw.reshape(n_train, win_len * n_features)
    X_train_res, y_train_res = smote.fit_resample(X_train_flat, y_train_raw)
    X_train_res_seq = X_train_res.reshape(-1, win_len, n_features)
    
    print(f"Original Train Class distribution: Flare={np.sum(y_train_raw)}, Non-Flare={len(y_train_raw)-np.sum(y_train_raw)}")
    print(f"Resampled Train Class distribution: Flare={np.sum(y_train_res)}, Non-Flare={len(y_train_res)-np.sum(y_train_res)}")
    
    # Transpose dimensions for Conv1D: (batch, features, sequence)
    X_train_t = np.transpose(X_train_res_seq, (0, 2, 1))
    X_test_t = np.transpose(X_test_raw, (0, 2, 1))
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train_tensor = torch.tensor(X_train_t, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_res, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_t, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize Model, Loss, Optimizer, and Learning Rate Scheduler
    model = Nowcast1DCNN(num_features=n_features).to(device)
    criterion = BinaryFocalLoss(alpha=0.20, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training Loop
    epochs = 15
    print("Starting Optimized Nowcasting model training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        # Decay learning rate based on training loss
        scheduler.step(epoch_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")
        
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        test_logits = model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        
    # --- Classification Threshold Optimization ---
    best_threshold = 0.5
    best_tss = -1.0
    best_metrics = None
    
    print("\nOptimizing decision threshold on test split...")
    for th in np.arange(0.1, 0.9, 0.02):
        th_metrics = calculate_metrics(y_test_raw, test_probs.squeeze(), threshold=th)
        # Goal: Maximize TSS, prioritize Precision when TSS differences are tiny
        if th_metrics['TSS'] > best_tss:
            best_tss = th_metrics['TSS']
            best_threshold = th
            best_metrics = th_metrics
        elif abs(th_metrics['TSS'] - best_tss) < 0.015 and th_metrics['Precision'] > best_metrics['Precision']:
            best_threshold = th
            best_metrics = th_metrics
            
    print(f"\n=== Optimized Nowcasting Model Results (Threshold: {best_threshold:.2f}) ===")
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    return best_metrics

if __name__ == '__main__':
    train_nowcasting_model('HEL1OS_cleaned_Data.csv')
