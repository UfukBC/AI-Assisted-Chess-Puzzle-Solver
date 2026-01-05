import cv2
import numpy as np
import mss
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

class BoardScanner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sct = mss.mss()
        self.region = {"top": START_Y, "left": START_X, "width": SQUARE_SIZE*8, "height": SQUARE_SIZE*8}
        
        # Ask player color once at startup
        while True:
            color_input = input("Are you playing white or black? (w/b): ").lower()
            if color_input in ['w', 'b']:
                self.playing_white = (color_input == 'w')
                break
            print("Invalid input. Please enter 'w' or 'b'")
        
        try:
            with open("classes.txt", "r") as f:
                self.classes = f.read().split(",")
            
            self.model = ChessNet(len(self.classes)).to(self.device)
            self.model.load_state_dict(torch.load("chess_model.pth", map_location=self.device))
            self.model.eval()
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Run train.py first!")
            exit()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.fen_map = {
            'bishop-white': 'B', 'bishop-black': 'b',
            'king-white': 'K', 'king-black': 'k',
            'knight-white': 'N', 'knight-black': 'n',
            'pawn-white': 'P', 'pawn-black': 'p',
            'queen-white': 'Q', 'queen-black': 'q',
            'rook-white': 'R', 'rook-black': 'r',
            'Empty': None
        }

    def scan_board(self):
        screen = np.array(self.sct.grab(self.region))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
        
        board_array = []
        for row in range(8):
            row_data = []
            for col in range(8):
                y, x = row * SQUARE_SIZE, col * SQUARE_SIZE
                sq_img = screen[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE]
                
                pil_img = Image.fromarray(sq_img)
                tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(tensor_img)
                    pred_idx = torch.argmax(output).item()
                    pred_class = self.classes[pred_idx]

                piece = self.fen_map.get(pred_class, None)
                row_data.append(piece)
            
            board_array.append(row_data)
        
        # If playing as black, flip the board 180 degrees
        if not self.playing_white:
            board_array = board_array[::-1]
            board_array = [row[::-1] for row in board_array]
        
        return board_array, 'w' if self.playing_white else 'b'
