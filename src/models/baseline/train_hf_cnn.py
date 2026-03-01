import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.models.baseline.hf_cnn_resnet import HFCNNResNet
from src.preprocessing.hf_dataloader import get_dataloaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = HFCNNResNet(num_classes=2).to(device)
processor = model.processor

# Load data
train_loader, val_loader = get_dataloaders(
    "splits/train.csv",
    "splits/val.csv",
    processor
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 7

best_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    save_path = "/content/data/AlertDrive/trained_model"
    os.makedirs(save_path, exist_ok=True)

    # Save every epoch
    torch.save(model.state_dict(), f"{save_path}/epoch_{epoch+1}.pth")

    # Best model based on validation loss
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        print("Best model updated based on validation loss!")