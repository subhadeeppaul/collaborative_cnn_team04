import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import json
from tqdm import tqdm

class CNNModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def get_loaders(train_dir, val_dir, batch_size=32):
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    train_set = datasets.ImageFolder(train_dir, transform=t)
    val_set = datasets.ImageFolder(val_dir, transform=t)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_v1(model, train_loader, val_loader, epochs=3, lr=0.001):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loss_last = 0
    val_loss_last = 0
    correct = 0
    total = 0

    for epoch in range(epochs):
        print("Epoch", epoch + 1, "/", epochs)
        model.train()
        epoch_loss = 0

        for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        train_loss_last = epoch_loss

        model.eval()
        vloss = 0
        correct = 0
        total = 0

        for imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
            out = model(imgs)
            loss = loss_fn(out, labels)
            vloss += loss.item()
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        val_loss_last = vloss

        print("Train Loss:", train_loss_last, "Val Loss:", val_loss_last)

    acc = correct / total if total > 0 else 0
    return {
        "train_loss_last": float(train_loss_last),
        "val_loss_last": float(val_loss_last),
        "val_accuracy": float(acc)
    }

def main():
    train_dir = "data/tongpython/train"
    val_dir = "data/tongpython/val"

    train_loader, val_loader = get_loaders(train_dir, val_dir)
    model = CNNModelV1()
    metrics = train_v1(model, train_loader, val_loader, epochs=3, lr=0.001)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model_v1.pth")

    os.makedirs("results", exist_ok=True)
    with open("results/metrics_v1.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
