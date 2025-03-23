import numpy as np
import random
import time

GRID_SIZE = 4
N_STEPS = 300                # Nombre maximal d'étapes dans la partie
MAX_MOVES_PER_ROLLOUT = 200  # Nombre maximal de mouvements par simulation Monte Carlo

# -------------------------------
# 1) Fonctions utilitaires du jeu
# -------------------------------
def init_board():
    board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    board = spawn_new_tile(board)
    board = spawn_new_tile(board)
    return board

def spawn_new_tile(board):
    empty_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if board[r, c] == 0]
    if not empty_cells:
        return board
    r, c = random.choice(empty_cells)
    board[r, c] = 2 if random.random() < 0.9 else 4
    return board

def compress_line(line):
    filtered = [x for x in line if x != 0]
    filtered += [0]*(GRID_SIZE - len(filtered))
    return filtered

def merge_line(line):
    score = 0
    for i in range(GRID_SIZE - 1):
        if line[i] != 0 and line[i] == line[i+1]:
            line[i] *= 2
            score += line[i]
            line[i+1] = 0
    return line, score

def move_left(board):
    new_board = np.zeros_like(board)
    total_score = 0
    for r in range(GRID_SIZE):
        compressed = compress_line(board[r, :])
        merged, sc = merge_line(compressed)
        merged = compress_line(merged)
        total_score += sc
        new_board[r, :] = merged
    return new_board, total_score

def move_right(board):
    flipped = np.fliplr(board)
    moved, score = move_left(flipped)
    moved_back = np.fliplr(moved)
    return moved_back, score

def move_up(board):
    transposed = board.T
    moved, score = move_left(transposed)
    moved_back = moved.T
    return moved_back, score

def move_down(board):
    transposed = board.T
    moved, score = move_right(transposed)
    moved_back = moved.T
    return moved_back, score

def get_possible_moves(board):
    moves = []
    for move_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
        new_board, _ = apply_move(board, move_name)
        if not np.array_equal(board, new_board):
            moves.append(move_name)
    return moves

def apply_move(board, move_name):
    if move_name == "UP":
        return move_up(board)
    elif move_name == "DOWN":
        return move_down(board)
    elif move_name == "LEFT":
        return move_left(board)
    elif move_name == "RIGHT":
        return move_right(board)
    else:
        return board, 0

def is_game_over(board):
    if 0 in board:
        return False
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE - 1):
            if board[r, c] == board[r, c+1]:
                return False
    for r in range(GRID_SIZE - 1):
        for c in range(GRID_SIZE):
            if board[r, c] == board[r+1, c]:
                return False
    return True

def get_score(board):
    return np.sum(board)

# -------------------------------
# 2) Monte Carlo simple sur CPU
# -------------------------------
def random_rollout_cpu(board, max_moves=MAX_MOVES_PER_ROLLOUT):
    b_copy = board.copy()
    for _ in range(max_moves):
        if is_game_over(b_copy):
            break
        moves = get_possible_moves(b_copy)
        if not moves:
            break
        chosen_move = random.choice(moves)
        b_copy, _ = apply_move(b_copy, chosen_move)
        b_copy = spawn_new_tile(b_copy)
    return get_score(b_copy)

def evaluate_move_cpu(board, move_name, n_rollouts=32):
    new_b, _ = apply_move(board, move_name)
    if np.array_equal(new_b, board):
        return -1
    new_b = spawn_new_tile(new_b)
    scores = []
    for _ in range(n_rollouts):
        s = random_rollout_cpu(new_b)
        scores.append(s)
    return np.mean(scores)

def choose_best_move_cpu(board, n_rollouts=32):
    moves = get_possible_moves(board)
    if not moves:
        return None
    best_score = -1
    best_move = None
    for m in moves:
        sc = evaluate_move_cpu(board, m, n_rollouts)
        if sc > best_score:
            best_score = sc
            best_move = m
    return best_move

def play_2048_cpu(n_steps=N_STEPS, rollouts=50):
    board = init_board()
    step = 0
    start_time = time.time()

    while step < n_steps and not is_game_over(board):
        move = choose_best_move_cpu(board, rollouts)
        if move is None:
            break
        board, _ = apply_move(board, move)
        board = spawn_new_tile(board)
        step += 1

        # Affichage live
        elapsed = time.time() - start_time
        current_score = get_score(board)
        possible_moves = get_possible_moves(board)
        print(f"[{step}/{n_steps}] Move: {move}, Score: {current_score}, "
              f"Moves left: {possible_moves}, Time: {elapsed:.2f}s")

    return board

# -------------------------------
# 3) Programme Principal CPU
# -------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=== 2048 CPU ===")
    start_t = time.time()
    final_board = play_2048_cpu()
    end_t = time.time()

    print("Partie terminée !")
    print(final_board)
    print(f"Score final = {get_score(final_board)}")
    print(f"Temps écoulé : {end_t - start_t:.2f} s")
