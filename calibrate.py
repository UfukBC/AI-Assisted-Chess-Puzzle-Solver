import pyautogui
import keyboard
import time

print("--- BOARD CALIBRATION TOOL ---")
print("1. Move your mouse to the TOP-LEFT corner of the chessboard (A8 square)")
print("2. Press 's' when you're ready")

while True:
    if keyboard.is_pressed('s'):
        start_x, start_y = pyautogui.position()
        print(f"\nSTART POSITION: X={start_x}, Y={start_y}")
        time.sleep(1)
        break

print("\n--------------------------------")
print("1. Now move your mouse to the BOTTOM-RIGHT corner of the chessboard (H1 square)")
print("2. Press 'e' when you're ready")

while True:
    if keyboard.is_pressed('e'):
        end_x, end_y = pyautogui.position()
        print(f"\nEND POSITION: X={end_x}, Y={end_y}")
        break

total_width = end_x - start_x
total_height = end_y - start_y

avg_total_size = (total_width + total_height) / 2
calculated_square_size = avg_total_size / 8

print("\n" + "="*30)
print("     RESULTS (Update board.py)     ")
print("="*30)
print(f"START_X     = {start_x}")
print(f"START_Y     = {start_y}")
print(f"SQUARE_SIZE = {int(calculated_square_size)}  (Exact: {calculated_square_size:.2f})")
print("="*30)

input("Press Enter to exit...")
