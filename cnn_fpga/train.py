"""
train.py — Train a CNN that exactly mirrors the FPGA hardware architecture.

Architecture:
    Input  : 28 x 28 x 1 greyscale
    Conv2D : 1 -> 1 channel, 3x3 kernel, padding=1    => 28 x 28 x 1
    ReLU
    MaxPool: 2x2, stride 2                            => 14 x 14 x 1
    Flatten                                           => 196
    Linear : 196 -> 10
    (Argmax at inference — no Softmax needed)

Learnable parameters:
    conv.weight  [1, 1, 3, 3]  =   9
    conv.bias    [1]            =   1
    fc.weight    [10, 196]      = 1960
    fc.bias      [10]           =  10
    ---------------------------------
    Total                        1980
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

DATA_DIR   = "./data"
MODEL_PATH = "./models/fpga_cnn.pth"
os.makedirs("./data",   exist_ok=True)
os.makedirs("./models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


class FPGACNN(nn.Module):
    """
    Direct software model of the FPGA CNN accelerator.

    Pipeline mirrors the RTL:
        slidingWindowAXI  — feeds 3x3 windows
        datapath_top      — Conv + bias + scale + ReLU
        max_pool_2d       — 2x2 max pool
        dense_layer       — 169 -> 10 linear
        argmax            — picks the highest score
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(196, 10, bias=True)

    def forward(self, x):
        x = self.conv(x)

        x = torch.relu(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=DATA_DIR, train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(
    root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=0)

print(f"[INFO] Train: {len(train_dataset)} images  |  Test: {len(test_dataset)} images")


model     = FPGACNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

total_params = sum(p.numel() for p in model.parameters())
print(f"[INFO] Total learnable parameters: {total_params}")
print(f"       conv.weight : {list(model.conv.weight.shape)}  = {model.conv.weight.numel()}")
print(f"       conv.bias   : {list(model.conv.bias.shape)}   = {model.conv.bias.numel()}")
print(f"       fc.weight   : {list(model.fc.weight.shape)} = {model.fc.weight.numel()}")
print(f"       fc.bias     : {list(model.fc.bias.shape)}  = {model.fc.bias.numel()}")


EPOCHS = 10
train_losses     = []
test_accuracies  = []

print("\n[INFO] Starting training...")
print(f"{'Epoch':>6} | {'Avg Loss':>10} | {'Test Acc (%)':>12}")
print("-" * 35)

for epoch in range(1, EPOCHS + 1):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total   += labels.size(0)

    accuracy = 100.0 * correct / total
    test_accuracies.append(accuracy)

    print(f"{epoch:>6} | {avg_loss:>10.4f} | {accuracy:>11.2f}%")


torch.save(model.state_dict(), MODEL_PATH)
print(f"\n[INFO] Model saved -> {MODEL_PATH}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS + 1), train_losses, marker='o', color='steelblue')
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.grid(True)

ax2.plot(range(1, EPOCHS + 1), test_accuracies, marker='o', color='darkorange')
ax2.set_title("Test Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("./models/training_curve.png", dpi=120)
print("[INFO] Training curve saved -> ./models/training_curve.png")
plt.show()

print("\n[INFO] Training complete.")
print(f"[INFO] Final test accuracy: {test_accuracies[-1]:.2f}%")
print("\n[NEXT STEP] Run extract_weights.py to pull out the FP32 parameters.")
