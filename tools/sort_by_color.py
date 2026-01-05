import cv2
import numpy as np
import os
import shutil

OLD_FOLDERS = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']
BASE_PATH = 'Dataset'

def detect_color(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    center_crop = img[h//3:2*h//3, w//3:2*w//3]
    gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    return 'white' if brightness > 100 else 'black'

print("Sorting pieces by color...")
print("=" * 50)

for piece_type in OLD_FOLDERS:
    old_folder = os.path.join(BASE_PATH, piece_type)
    
    if not os.path.exists(old_folder):
        print(f"Skipping {piece_type} (folder not found)")
        continue
    
    files = [f for f in os.listdir(old_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print(f"Skipping {piece_type} (no images)")
        continue
    
    print(f"\nProcessing {piece_type}: {len(files)} images")
    
    white_count = 0
    black_count = 0
    
    for filename in files:
        src_path = os.path.join(old_folder, filename)
        color = detect_color(src_path)
        
        if color is None:
            print(f"  ERROR: Could not read {filename}")
            continue
        
        dest_folder = os.path.join(BASE_PATH, f"{piece_type}-{color}")
        dest_path = os.path.join(dest_folder, filename)
        
        shutil.copy2(src_path, dest_path)
        
        if color == 'white':
            white_count += 1
        else:
            black_count += 1
    
    print(f"  White: {white_count}, Black: {black_count}")

print("\n" + "=" * 50)
print("Done! Check the folders and fix any mistakes manually.")
print("After fixing, run: python train.py")
