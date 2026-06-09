import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

DATA_DIR = '/deck/users/mpiras/dataset/hsmr'

CLASS_PATTERNS = [
    ('butt_wink',         f'{DATA_DIR}/HSMR-squat_butt_wink*.npy'),
    ('depth_high',        f'{DATA_DIR}/HSMR-squat_depth_high*.npy'),
    ('hands_wide',        f'{DATA_DIR}/HSMR-squat_hands_wide*.npy'),
    ('head_position',     f'{DATA_DIR}/HSMR-squat_head_position*.npy'),
    ('high_heel',         f'{DATA_DIR}/HSMR-squat_high_heel*.npy'),
    ('no_errors',         f'{DATA_DIR}/HSMR-squat_no_errors*.npy'),
    ('rdl_hands_forward', f'{DATA_DIR}/HSMR-rdl_hands_forward*.npy'),
    ('rdl_no_error',      f'{DATA_DIR}/HSMR-rdl_no_error*.npy'),
    ('rdl_too_much_depth',f'{DATA_DIR}/HSMR-rdl_too_much_depth*.npy'),
]
CLASS_NAMES = [name for name, _ in CLASS_PATTERNS]
NUM_CLASSES = len(CLASS_PATTERNS)


def load_class(pattern, label):
    files = glob.glob(pattern)
    samples = []
    for file in files:
        arr = np.load(file, allow_pickle=True)
        tokens = [frame['poses'].flatten() for frame in arr]
        samples.append({'tokens': tokens, 'label': label, 'filename': file})
    print(f"  label {label} ({CLASS_NAMES[label]}): {len(files)} files")
    return samples


print("Loading dataset...")
dataset = []
for label, (name, pattern) in enumerate(CLASS_PATTERNS):
    dataset.extend(load_class(pattern, label))
print(f"Total samples: {len(dataset)}")

# Pad/truncate to uniform sequence length
max_seq_length = max(len(s['tokens']) for s in dataset)
print(f"Max sequence length: {max_seq_length}")
for s in dataset:
    tokens = s['tokens']
    if len(tokens) < max_seq_length:
        padding = [np.zeros_like(tokens[0]) for _ in range(max_seq_length - len(tokens))]
        s['tokens'] = tokens + padding
    else:
        s['tokens'] = tokens[:max_seq_length]

X = np.array([s['tokens'] for s in dataset])
y = np.array([s['label'] for s in dataset])
print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32, shuffle=False)


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


input_dim = X_train_t.shape[-1]
print(f"Input dim: {input_dim}, Seq len: {X_train_t.shape[1]}, Classes: {NUM_CLASSES}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = TransformerClassifier(input_dim, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        _, predicted = torch.max(model(X_batch), 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
print(f"Test Accuracy: {correct / total:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_t.to(device))
    _, y_pred = torch.max(outputs, 1)
    y_pred = y_pred.cpu().numpy()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap='Blues', aspect='auto')
plt.colorbar()
plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha='right')
plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show(block=True)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

print("\nPer-Class Accuracy:")
per_class_acc = []
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    correct = (y_pred[mask] == y_test[mask]).sum()
    total = mask.sum()
    acc = correct / total if total > 0 else 0.0
    per_class_acc.append(acc)
    print(f"  {name:<22} {correct:>3}/{total:<3}  ({acc:.1%})")

plt.figure(figsize=(10, 5))
bars = plt.barh(CLASS_NAMES, [a * 100 for a in per_class_acc], color='steelblue')
plt.axvline(x=100, color='gray', linestyle='--', linewidth=0.8)
for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{acc:.1%}", va='center', fontsize=9)
plt.xlabel("Accuracy (%)")
plt.title("Per-Class Accuracy")
plt.xlim(0, 110)
plt.tight_layout()
plt.savefig("per_class_accuracy.png", dpi=300)
plt.show(block=True)

# ----------------------------------------------------------
# Per-class average saliency
# For each test sample, compute gradient saliency w.r.t. its
# true class, average over frames, then average per class.
# Result: which pose parameters drive each class prediction.
# ----------------------------------------------------------

POSE_NAMES = [
    "Pelvis Tilt", "Pelvis List", "Pelvis Rot",
    "R Hip Flexion", "R Hip Adduction", "R Hip Rotation",
    "R Knee Angle", "R Ankle Angle", "R Subtalar", "R MTP",
    "L Hip Flexion", "L Hip Adduction", "L Hip Rotation",
    "L Knee Angle", "L Ankle Angle", "L Subtalar", "L MTP",
    "Lumbar Bend", "Lumbar Ext", "Lumbar Twist",
    "Thorax Bend", "Thorax Ext", "Thorax Twist",
    "Head Bend", "Head Ext", "Head Twist",
    "R Scap Abd", "R Scap Elev", "R Scap Up-Rot",
    "R Shoulder X", "R Shoulder Y", "R Shoulder Z",
    "R Elbow Flex", "R Forearm Pro/Sup", "R Wrist Flex", "R Wrist Dev",
    "L Scap Abd", "L Scap Elev", "L Scap Up-Rot",
    "L Shoulder X", "L Shoulder Y", "L Shoulder Z",
    "L Elbow Flex", "L Forearm Pro/Sup", "L Wrist Flex", "L Wrist Dev",
]

def saliency_per_frame(model, x, target_class):
    """Returns (seq_len, n_features) normalised absolute gradient saliency."""
    model.eval()
    inp = x.clone().detach().to(device).requires_grad_(True)
    score = model(inp)[0, target_class]
    model.zero_grad()
    score.backward()
    slc = inp.grad.data.abs().squeeze().cpu().numpy()
    slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
    return slc

def top_attended_frames(model, x, k=10):
    """Returns indices of the k frames with highest average attention."""
    model.eval()
    x_proj = model.input_projection(x)
    last_layer = model.transformer_encoder.layers[-1]
    _, attn = last_layer.self_attn(x_proj, x_proj, x_proj, need_weights=True)
    importance = attn.detach().cpu().numpy()[0].mean(axis=0)
    return np.argsort(importance)[::-1][:k]

TOP_K = 10
THRESHOLD = 0.5

print("Computing per-class saliency on test set...")
topk_saliency  = {i: [] for i in range(NUM_CLASSES)}  # approach 1
thresh_freq    = {i: [] for i in range(NUM_CLASSES)}  # approach 2

for i in range(len(X_test)):
    if y_pred[i] != y_test[i]:
        continue
    x = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(device)
    label = y_test[i]
    slc = saliency_per_frame(model, x, label)          # (seq_len, n_features)

    # Approach 1: mean saliency of top-K attended frames
    top_idx = top_attended_frames(model, x, k=TOP_K)
    topk_saliency[label].append(slc[top_idx].mean(axis=0))

    # Approach 2: fraction of frames each feature exceeds threshold
    thresh_freq[label].append((slc > THRESHOLD).mean(axis=0))

# (num_classes, n_features)
avg_topk  = np.array([np.mean(topk_saliency[i], axis=0) for i in range(NUM_CLASSES)])
avg_freq  = np.array([np.mean(thresh_freq[i],   axis=0) for i in range(NUM_CLASSES)])

def plot_saliency_heatmap(matrix, title, filename, normalise=True):
    if normalise:
        matrix = matrix / (matrix.max(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(20, 6))
    im = ax.imshow(matrix, aspect='auto', cmap='hot', vmin=0, vmax=1)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xticks(range(len(POSE_NAMES)))
    ax.set_xticklabels(POSE_NAMES, rotation=90, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show(block=True)

plot_saliency_heatmap(
    avg_topk,
    f"Saliency on Top-{TOP_K} Attended Frames per Class (normalised within class)",
    "saliency_topk_frames.png"
)

plot_saliency_heatmap(
    avg_freq,
    f"Feature Activation Frequency above threshold={THRESHOLD} per Class",
    "saliency_threshold_freq.png",
    normalise=False  # raw frequency (0-1) is already interpretable
)

print(f"\nTop-5 pose parameters per class (top-{TOP_K} frames):")
for i, name in enumerate(CLASS_NAMES):
    top5 = np.argsort(avg_topk[i])[::-1][:5]
    params = ", ".join(f"{POSE_NAMES[j]} ({avg_topk[i][j]:.3f})" for j in top5)
    print(f"  {name}: {params}")

print(f"\nTop-5 pose parameters per class (threshold frequency):")
for i, name in enumerate(CLASS_NAMES):
    top5 = np.argsort(avg_freq[i])[::-1][:5]
    params = ", ".join(f"{POSE_NAMES[j]} ({avg_freq[i][j]:.2%})" for j in top5)
    print(f"  {name}: {params}")
