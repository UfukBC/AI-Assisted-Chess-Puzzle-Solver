from stockfish import Stockfish

STOCKFISH_PATH = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"

class ChessEngine:
    def __init__(self):
        try:
            self.engine = Stockfish(path=STOCKFISH_PATH)
            self.engine.set_depth(20)
            print("Stockfish loaded.")
        except Exception as e:
            print(f"Stockfish error: {e}")
            print("Check the path!")
            exit()

    def board_to_fen(self, board_array, active_color='w'):
        fen_rows = []
        for row in board_array:
            fen_line = ""
            empty_count = 0
            for piece in row:
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_line += str(empty_count)
                        empty_count = 0
                    fen_line += piece
            
            if empty_count > 0:
                fen_line += str(empty_count)
            fen_rows.append(fen_line)
        
        return "/".join(fen_rows) + f" {active_color} - - 0 1"

    def is_valid_fen(self, fen):
        # Check if FEN has both kings
        if 'K' not in fen or 'k' not in fen:
            return False
        
        # Check for impossible pawn positions
        rows = fen.split(' ')[0].split('/')
        
        # 1st and 8th rank cannot have pawns (indices 0 and 7)
        if 'P' in rows[0] or 'p' in rows[0] or 'P' in rows[7] or 'p' in rows[7]:
            return False
        
        return True

    def get_best_move(self, fen):
        if not self.is_valid_fen(fen):
            return None
            
        try:
            self.engine.set_fen_position(fen)
            return self.engine.get_best_move()
        except Exception as e:
            print(f"Stockfish error: {e}")
            return None

    def get_best_line(self, fen, move_count=5, filter_color=None):
        try:
            self.engine.set_fen_position(fen)
            all_moves = []
            
            for i in range(move_count * 2):
                best_move = self.engine.get_best_move()
                if not best_move:
                    break
                
                all_moves.append(best_move)
                self.engine.make_moves_from_current_position([best_move])
            
            # If filter_color is set, return only moves for that color
            if filter_color:
                # Active color (whose turn it is) makes moves at 0,2,4...
                # So if we want active color's moves, take even indices
                return all_moves[0::2][:move_count]
            
            return all_moves[:move_count]
        except Exception as e:
            print(f"Stockfish error: {e}")
            return []
