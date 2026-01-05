import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

DATA_PATH = "Dataset/"
MODEL_SAVE_PATH = "chess_model.pth"
EPOCHS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

class ChessNet(nn.Module):
    def __init__(self, num_classes):
        super(ChessNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
    except:
        print("ERROR: Dataset folder not found!")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    class_names = dataset.classes
    print(f"Detected classes: {class_names}")
    with open("classes.txt", "w") as f:
        f.write(",".join(class_names))

    model = ChessNet(len(class_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training started...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} completed. Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Done! Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
