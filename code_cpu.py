import numpy as np
import random
import time

GRID_SIZE = 4

# -------------------------------
# 1) Fonctions utilitaires du jeu
# -------------------------------
def init_board():
    """ Initialise le plateau 4x4 et ajoute 2 tuiles. """
    board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    board = spawn_new_tile(board)
    board = spawn_new_tile(board)
    return board

def spawn_new_tile(board):
    """ Fait apparaître une nouvelle tuile (2 ou 4) dans une case vide au hasard. """
    empty_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if board[r, c] == 0]
    if not empty_cells:
        return board  # aucune place
    r, c = random.choice(empty_cells)
    board[r, c] = 2 if random.random() < 0.9 else 4
    return board

def compress_line(line):
    """
    Déplace toutes les tuiles non nulles d'une ligne à gauche (sans fusion).
    Ex: [2, 0, 2, 4] -> [2, 2, 4, 0]
    """
    filtered = [x for x in line if x != 0]
    filtered += [0]*(GRID_SIZE - len(filtered))
    return filtered

def merge_line(line):
    """
    Fusionne les tuiles adjacentes (de gauche à droite).
    Ex: [2, 2, 4, 4] -> [4, 0, 8, 0]
    Retourne (nouvelle_ligne, score_gagné)
    """
    score = 0
    for i in range(GRID_SIZE - 1):
        if line[i] != 0 and line[i] == line[i+1]:
            line[i] *= 2
            score += line[i]
            line[i+1] = 0
    return line, score

def move_left(board):
    """
    Effectue un mouvement "LEFT" sur tout le plateau.
    Retourne (nouveau_plateau, score_gagné).
    """
    new_board = np.zeros_like(board)
    total_score = 0
    for r in range(GRID_SIZE):
        compressed = compress_line(board[r, :])
        merged, sc = merge_line(compressed)
        merged = compress_line(merged)  # re-compress après fusion
        total_score += sc
        new_board[r, :] = merged
    return new_board, total_score

def move_right(board):
    """
    Mouvement "RIGHT".
    On inverse, on move_left, puis on ré-inverse.
    """
    flipped = np.fliplr(board)
    moved, score = move_left(flipped)
    moved_back = np.fliplr(moved)
    return moved_back, score

def move_up(board):
    """
    Mouvement "UP".
    On transpose, on move_left, puis on re-transpose.
    """
    transposed = board.T
    moved, score = move_left(transposed)
    moved_back = moved.T
    return moved_back, score

def move_down(board):
    """
    Mouvement "DOWN".
    On transpose, on move_right, puis on re-transpose.
    """
    transposed = board.T
    moved, score = move_right(transposed)
    moved_back = moved.T
    return moved_back, score

def get_possible_moves(board):
    """
    Retourne la liste des coups valides (strings) pour le plateau donné,
    càd ceux qui modifient effectivement l'état.
    """
    moves = []
    for move_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
        new_board, _ = apply_move(board, move_name)
        if not np.array_equal(board, new_board):
            moves.append(move_name)
    return moves

def apply_move(board, move_name):
    """
    Applique le move 'UP', 'DOWN', 'LEFT' ou 'RIGHT' sur 'board'.
    Retourne (nouveau_board, score_gagné).
    """
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
    """ Renvoie True si plus aucun coup n'est possible. """
    if 0 in board:
        return False
    # check si un merge est encore possible
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE-1):
            if board[r, c] == board[r, c+1]:
                return False
    for r in range(GRID_SIZE-1):
        for c in range(GRID_SIZE):
            if board[r, c] == board[r+1, c]:
                return False
    return True

def get_score(board):
    """ Score = somme de toutes les tuiles """
    return np.sum(board)

# -------------------------------
# 2) Monte Carlo simple sur CPU
# -------------------------------
def random_rollout_cpu(board, max_moves=200):
    """
    Joue au hasard depuis l'état 'board' jusqu'à la fin (ou max_moves).
    Retourne le score final.
    """
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
    """
    Évalue un coup en jouant n_rollouts parties aléatoires à partir de board 
    (après avoir appliqué le coup).
    Retourne le score moyen obtenu.
    """
    new_b, _ = apply_move(board, move_name)
    # Si le coup ne change rien, on renvoie un score moyen négatif
    if np.array_equal(new_b, board):
        return -1
    # On spawn une tuile
    new_b = spawn_new_tile(new_b)

    # Monte Carlo rollouts
    scores = []
    for _ in range(n_rollouts):
        s = random_rollout_cpu(new_b)
        scores.append(s)
    return np.mean(scores)

def choose_best_move_cpu(board, n_rollouts=32):
    """
    Choisit le coup avec la meilleure moyenne de score final (Monte Carlo).
    """
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

def play_2048_cpu(n_steps=300, rollouts=50):
    """
    Joue n_steps coups en choisissant à chaque fois le meilleur coup (Monte Carlo).
    """
    board = init_board()
    step = 0
    while step < n_steps and not is_game_over(board):
        move = choose_best_move_cpu(board, rollouts)
        if move is None:
            break
        board, _ = apply_move(board, move)
        board = spawn_new_tile(board)
        step += 1
    return board

# -------------------------------
# 3) Programme Principal CPU
# -------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=== 2048 AI CPU ===")
    start_t = time.time()
    final_board = play_2048_cpu(n_steps=300, rollouts=50)
    end_t = time.time()

    print("Partie terminée !")
    print(final_board)
    print(f"Score final = {get_score(final_board)}")
    print(f"Temps écoulé : {end_t - start_t:.2f} s")
