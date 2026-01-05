import cv2
import numpy as np
import mss
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

START_X = 237
START_Y = 136
SQUARE_SIZE = 110

class ChessNet(nn.Module):
    def __init__(self, num_classes):
        super(ChessNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16 * 16, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("classes.txt", "r") as f:
    classes = f.read().split(",")

model = ChessNet(len(classes)).to(device)
model.load_state_dict(torch.load("chess_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

sct = mss.mss()
region = {"top": START_Y, "left": START_X, "width": SQUARE_SIZE*8, "height": SQUARE_SIZE*8}
screen = np.array(sct.grab(region))
screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

os.makedirs("debug_squares", exist_ok=True)

print("CHESS BOARD ANALYSIS\n")
print(f"Detected classes: {classes}\n")
print("=" * 70)

for row in range(8):
    for col in range(8):
        y, x = row * SQUARE_SIZE, col * SQUARE_SIZE
        sq_img = screen[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE]
        
        pil_img = Image.fromarray(sq_img)
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor_img)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(output).item()
            pred_class = classes[pred_idx]
            confidence = probs[pred_idx].item() * 100
        
        h, w = sq_img.shape[:2]
        center_crop = sq_img[h//3:2*h//3, w//3:2*w//3]
        gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        color = 'White' if brightness > 100 else 'Black'
        
        square_color = "L" if (row + col) % 2 == 0 else "D"
        
        square_name = f"{chr(97+col)}{8-row}"
        cv2.imwrite(f"debug_squares/{square_name}_{pred_class}.png", cv2.cvtColor(sq_img, cv2.COLOR_RGB2BGR))
        
        print(f"{square_color} {square_name}: {pred_class:8s} ({confidence:5.1f}%) | Color: {color:5s} | Brightness: {brightness:.0f}")

print("=" * 70)
print(f"\nAll squares saved to 'debug_squares/' folder.")
print("Check images to verify coordinates are correct.")
