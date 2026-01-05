# Chess Piece Identifier

AI-powered chess vision bot that analyzes a chessboard from screen capture and suggests best moves using Stockfish engine.

## Features

- Real-time chess position detection using CNN
- Automatic board scanning and FEN generation
- Integration with Stockfish for move analysis
- Support for both white and black perspectives
- 5-move deep analysis

## Requirements

- Python 3.8+
- 1920x1080 display resolution (recalibration needed for other resolutions)
- Stockfish chess engine

## Installation

### Quick Start (Using Pre-trained Model)

If you want to use the pre-trained model immediately:

1. **Install Dependencies**
   ```bash
   pip install torch torchvision opencv-python mss pillow stockfish keyboard winsound
   ```

2. **Download Stockfish**
   
   Download from [stockfishchess.org](https://stockfishchess.org/download/) and update `moves.py`:
   ```python
   STOCKFISH_PATH = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
   ```

3. **Calibrate for Your Display** (if not using 1920x1080)
   ```bash
   python calibrate.py
   ```

4. **Run the Bot**
   ```bash
   python main.py
   ```

### Training Your Own Model

If you want to train from scratch:

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python mss pillow stockfish keyboard winsound
```

### 2. Download Stockfish

Download Stockfish from [stockfishchess.org](https://stockfishchess.org/download/) and update the path in `moves.py`:

```python
STOCKFISH_PATH = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
```

### 3. Dataset Preparation

Download the chess piece dataset from [Kaggle Chess Pieces Dataset](https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset) or use your own images.

Organize the dataset in the following structure:

```
Dataset/
  bishop-white/
  bishop-black/
  king-white/
  king-black/
  knight-white/
  knight-black/
  pawn-white/
  pawn-black/
  queen-white/
  queen-black/
  rook-white/
  rook-black/
  Empty/
```

**Important:** Each piece type should have separate folders for white and black pieces to ensure accurate color detection.

### 4. Train the Model

```bash
python train.py
```

This will:
- Process images from the Dataset folder
- Train a CNN model for 15 epochs
- Save the model as `chess_model.pth`
- Generate `classes.txt` with detected classes

## Display Calibration

**Note:** Default coordinates are configured for 1920x1080 resolution. For other resolutions, you must recalibrate.

### Calibration Steps:

1. Open your chess application in fullscreen
2. Run the calibration tool:
   ```bash
   python calibrate.py
   ```
3. Follow the on-screen instructions:
   - Press 'S' when mouse is at A8 square (top-left)
   - Press 'E' when mouse is at H1 square (bottom-right)
4. Update the values in `board.py`:
   ```python
   START_X = <your_value>
   START_Y = <your_value>
   SQUARE_SIZE = <your_value>
   ```

## Usage

### Run the Chess Bot

```bash
python main.py
```

1. Choose your color when prompted (w/b)
2. Set up a chess position on your board
3. Wait for analysis (you'll hear a beep when complete)
4. Follow the suggested moves

### Output Example

```
Are you playing white or black? (w/b): w
Model loaded.
Stockfish loaded.
System ready. Starting analysis...

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1
YOUR BEST MOVES: e2e4 -> g1f3 -> f1c4 -> d2d4 -> b1c3

Analysis complete.
```

## File Structure

### Core Files
- `main.py` - Main application entry point
- `board.py` - Board scanning and AI model
- `moves.py` - Stockfish integration and move calculation
- `train.py` - Model training script

### Tools (for dataset preparation)
- `tools/capture_empty.py` - Capture empty square images
- `tools/sort_by_color.py` - Auto-sort pieces by color
- `tools/test_debug.py` - Debug board detection and save squares

### Configuration
- `calibrate.py` - Screen calibration tool
- `classes.txt` - Auto-generated class names
- `chess_model.pth` - Trained model weights

## Training Your Own Model

If you want to use a different dataset:

1. **Organize your images** in the Dataset folder structure (13 classes: 12 piece types + Empty)

2. **Use training tools** (optional):
   ```bash
   # Capture empty squares for training
   python tools/capture_empty.py
   
   # Auto-sort pieces by color (check results manually)
   python tools/sort_by_color.py
   ```

3. **Train the model:**
   ```bash
   python train.py
   ```

4. **Test detection:**
   ```bash
   python tools/test_debug.py
   ```
   Check `debug_squares/` folder to verify piece detection accuracy.

## Troubleshooting

### "Stockfish process has crashed"
- FEN string is invalid (missing kings or invalid position)
- Check if both kings are detected on the board

### Model detects wrong pieces
- Retrain with more diverse dataset
- Ensure correct color separation (white/black folders)
- Check calibration coordinates

### Wrong perspective
- Make sure you selected the correct color (w/b) when prompted
- The board will automatically flip for black perspective

## Credits

Built with PyTorch, OpenCV, and Stockfish chess engine.

## License

MIT License

