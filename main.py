import time
import winsound
from board import BoardScanner
from moves import ChessEngine

def main():
    print("Chess Vision Bot")
    print("="*50)
    
    scanner = BoardScanner()
    engine = ChessEngine()
    
    print("="*50)
    print("System ready. Starting analysis...")
    print("Press Ctrl+C to stop")
    print()
    
    last_fen = ""
    
    while True:
        try:
            board, active_color = scanner.scan_board()
            fen = engine.board_to_fen(board, active_color)
            
            if fen != last_fen:
                print(f"\nFEN: {fen}")
                
                if len(fen.split('/')) == 8 and 'K' in fen and 'k' in fen:
                    best_line = engine.get_best_line(fen, move_count=5, filter_color=active_color)
                    if best_line:
                        print(f"YOUR BEST MOVES: {' -> '.join(best_line)}")
                        print("\nAnalysis complete.")
                        winsound.Beep(1000, 200)
                        break
                    else:
                        print("Could not calculate moves")
                else:
                    print("Invalid FEN - waiting for valid position (need both kings)...")
                
                last_fen = fen
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n\nStopping...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
