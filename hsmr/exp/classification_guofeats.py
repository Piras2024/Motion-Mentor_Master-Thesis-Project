import numpy as np
import glob

DATASET_DIR = '/deck/users/mpiras/dataset/squat_micc_guofeats'

def load_category(pattern, label):
    files = glob.glob(f'{DATASET_DIR}/{pattern}')
    samples = []
    for file in files:
        arr = np.load(file, allow_pickle=True)  # shape: (T, 263)
        tokens = [arr[i] for i in range(len(arr))]
        samples.append({'tokens': tokens, 'label': label, 'filename': file})
    print(f"Loaded {len(files)} files for pattern {pattern}")
    return samples

butt_wink_dataset   = load_category('HSMR-squat_butt_wink*_guofeats.npy', 0)
depth_high_dataset  = load_category('HSMR-squat_depth_high*_guofeats.npy', 1)
hands_wide_dataset  = load_category('HSMR-squat_hands_wide*_guofeats.npy', 2)
high_heels_dataset  = load_category('HSMR-high_heel*_guofeats.npy', 3)
head_position_dataset = load_category('HSMR-head_position*_guofeats.npy', 4)
no_errors_dataset   = load_category('HSMR-no_errors*_guofeats.npy', 5)

dataset = (butt_wink_dataset + depth_high_dataset + hands_wide_dataset +
           high_heels_dataset + head_position_dataset + no_errors_dataset)
print(f"Total samples in dataset: {len(dataset)}")

# pad/truncate to max sequence length
max_seq_length = max(len(s['tokens']) for s in dataset)
print(f"Max sequence length: {max_seq_length}")
for sample in dataset:
    tokens = sample['tokens']
    if len(tokens) < max_seq_length:
        padding = [np.zeros_like(tokens[0]) for _ in range(max_seq_length - len(tokens))]
        sample['tokens'] = tokens + padding
    else:
        sample['tokens'] = tokens[:max_seq_length]

X = np.array([s['tokens'] for s in dataset])
y = np.array([s['label'] for s in dataset])
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

print(f"X_train_tensor shape: {X_train_tensor.shape}")
print(f"X_test_tensor shape:  {X_test_tensor.shape}")

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_tensor,  y_test_tensor),  batch_size=32, shuffle=False)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


input_dim  = X_train_tensor.shape[-1]
num_classes = len(set(y_train.tolist()))
print(f"Input dimension: {input_dim}, Num classes: {num_classes}")

model = TransformerClassifier(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        _, predicted = torch.max(model(X_batch), 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
print(f"Test Accuracy: {correct/total:.4f}")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    _, y_pred = torch.max(model(X_test_tensor), 1)
    y_pred = y_pred.numpy()

CLASS_NAMES = ['Butt Wink', 'Depth High', 'Hands Wide', 'High Heels', 'Head Position', 'No Errors']
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues', aspect='auto')
plt.colorbar()
plt.xticks(range(6), CLASS_NAMES, rotation=45, ha='right')
plt.yticks(range(6), CLASS_NAMES)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (Guo Features)')
for i in range(6):
    for j in range(6):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig("confusion_matrix_guofeats.png", dpi=300)
plt.show(block=True)

print("\nPer-Class Accuracy and Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
