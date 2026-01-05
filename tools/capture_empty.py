import cv2
import numpy as np
import mss
import os
import keyboard
import time

START_X = 237
START_Y = 136
SQUARE_SIZE = 110

# Top-left 1st, 2nd and 3rd squares (a8, b8, c8)
SQUARES = [
    (0, 0),  # 1st square (column 0, row 0) - a8
    (1, 0),  # 2nd square (column 1, row 0) - b8
    (2, 0)   # 3rd square (column 2, row 0) - c8
]

OUTPUT_DIR = "Dataset/Empty"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sct = mss.mss()
photo_count = 0

print("EMPTY SQUARE CAPTURE TOOL")
print("=" * 50)
print("Keep chess board open")
print("Make sure 1st, 2nd and 3rd squares are empty (a8, b8, c8)")
print("Press Q = Take photo")
print("Press ESC = Exit")
print("=" * 50)
print(f"Photos will be saved to: {OUTPUT_DIR}\n")
print("Ready! Press Q to start...\n")

while True:
    try:
        if keyboard.is_pressed('q'):
            region = {"top": START_Y, "left": START_X, "width": SQUARE_SIZE*8, "height": SQUARE_SIZE*8}
            screen = np.array(sct.grab(region))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
            
            timestamp = int(time.time())
            
            for col, row in SQUARES:
                y, x = row * SQUARE_SIZE, col * SQUARE_SIZE
                sq_img = screen[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE]
                
                square_name = f"{chr(97+col)}{8-row}"
                filename = f"{OUTPUT_DIR}/empty_{square_name}_{timestamp}_{photo_count}.png"
                cv2.imwrite(filename, sq_img)
                print(f"Saved {square_name}: {filename}")
                
                photo_count += 1
            
            print(f"Total {photo_count} photos taken\n")
            time.sleep(0.3)
        
        if keyboard.is_pressed('esc'):
            print("\nExiting...")
            break
        
        time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\nStopped by user...")
        break

print(f"\nDone! Total {photo_count} photos saved to '{OUTPUT_DIR}'.")
