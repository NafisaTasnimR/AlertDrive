import torch
import torch.nn as nn
import torch.optim as optim

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

num_epochs = 1

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

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "trained_models/hf_cnn_resnet.pth")