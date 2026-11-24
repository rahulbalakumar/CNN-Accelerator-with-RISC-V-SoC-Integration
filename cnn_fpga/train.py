"""
train.py - Train the FP32 CNN that mirrors the FPGA hardware.

Architecture (see fpga_model.py):
    28x28 -> Conv3x3 (valid, 1 ch) -> ReLU -> MaxPool 2x2 -> 169 -> Linear -> 10

Learnable parameters:
    conv.weight [1,1,3,3] =    9
    conv.bias   [1]       =    1
    fc.weight   [10,169]  = 1690
    fc.bias     [10]      =   10
    ----------------------------
    Total                   1710

Run from the cnn_fpga folder:   python train.py
Next step:                      python quantize.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fpga_model import FPGACNN, load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "fpga_cnn.pth")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

EPOCHS = 15
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


def to_tensor(images):
    # Same scaling as the hardware sees: raw byte / 255. No mean/std normalisation,
    # so the int8 model can use the raw 0..255 pixel directly.
    return torch.from_numpy(images).float().unsqueeze(1) / 255.0


train_x, train_y = load_mnist("train")
test_x, test_y = load_mnist("test")
train_loader = DataLoader(TensorDataset(to_tensor(train_x), torch.from_numpy(train_y).long()),
                          batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(to_tensor(test_x), torch.from_numpy(test_y).long()),
                         batch_size=1000, shuffle=False)
print(f"[INFO] Train: {len(train_x)} images  |  Test: {len(test_x)} images")

model = FPGACNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

print(f"[INFO] Total learnable parameters: {sum(p.numel() for p in model.parameters())}")

train_losses, test_accuracies = [], []
print(f"\n{'Epoch':>6} | {'Avg Loss':>10} | {'Test Acc (%)':>12}")
print("-" * 35)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    scheduler.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = model(images.to(device)).argmax(dim=1)
            correct += (pred == labels.to(device)).sum().item()
    acc = 100.0 * correct / len(test_x)
    train_losses.append(running / len(train_loader))
    test_accuracies.append(acc)
    print(f"{epoch:>6} | {train_losses[-1]:>10.4f} | {acc:>11.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print(f"\n[INFO] Model saved -> {MODEL_PATH}")
print(f"[INFO] Final FP32 test accuracy: {test_accuracies[-1]:.2f}%")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, EPOCHS + 1), train_losses, marker="o")
    ax1.set_title("Training loss"); ax1.set_xlabel("Epoch"); ax1.grid(True)
    ax2.plot(range(1, EPOCHS + 1), test_accuracies, marker="o")
    ax2.set_title("Test accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "models", "training_curve.png"), dpi=120)
    print("[INFO] Training curve saved -> models/training_curve.png")
except ImportError:
    pass

print("\n[NEXT STEP] python quantize.py")
