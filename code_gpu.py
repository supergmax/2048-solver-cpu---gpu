import numpy as np
import random
import time
from numba import cuda, njit, int32  # int32 reste utile pour cuda.local.array
# Mais pour device_array, on utilisera np.int32

GRID_SIZE = 4

@njit
def compress_line(line):
    filtered = [x for x in line if x != 0]
    filtered += [0]*(GRID_SIZE - len(filtered))
    return filtered

@njit
def merge_line(line):
    score = 0
    for i in range(GRID_SIZE - 1):
        if line[i] != 0 and line[i] == line[i+1]:
            line[i] *= 2
            score += line[i]
            line[i+1] = 0
    return line, score

@njit
def move_left_njit(board):
    new_board = np.zeros_like(board)
    total_score = 0
    for r in range(GRID_SIZE):
        row = board[r, :]
        comp = compress_line(row)
        merged, sc = merge_line(comp)
        comp2 = compress_line(merged)
        new_board[r, :] = comp2
        total_score += sc
    return new_board, total_score

@njit
def move_right_njit(board):
    flipped = np.fliplr(board)
    moved, score = move_left_njit(flipped)
    moved_back = np.fliplr(moved)
    return moved_back, score

@njit
def move_up_njit(board):
    transposed = board.T
    moved, score = move_left_njit(transposed)
    return moved.T, score

@njit
def move_down_njit(board):
    transposed = board.T
    moved, score = move_right_njit(transposed)
    return moved.T, score

@njit
def apply_move_njit(board, move_id):
    if move_id == 0:
        return move_up_njit(board)
    elif move_id == 1:
        return move_down_njit(board)
    elif move_id == 2:
        return move_left_njit(board)
    else:
        return move_right_njit(board)

@njit
def board_equals(b1, b2):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if b1[r,c] != b2[r,c]:
                return False
    return True

@njit
def get_score_njit(board):
    return np.sum(board)

def spawn_new_tile(board):
    empty_cells = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board[r,c] == 0:
                empty_cells.append((r,c))
    if not empty_cells:
        return board
    rr, cc = random.choice(empty_cells)
    board[rr, cc] = 2 if random.random() < 0.9 else 4
    return board

def get_possible_moves(board):
    possible = []
    for move_id in [0,1,2,3]:
        new_b, _ = apply_move_njit(board, move_id)
        if not board_equals(new_b, board):
            possible.append(move_id)
    return possible

def is_game_over(board):
    if 0 in board:
        return False
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE-1):
            if board[r,c] == board[r,c+1]:
                return False
    for r in range(GRID_SIZE-1):
        for c in range(GRID_SIZE):
            if board[r,c] == board[r+1,c]:
                return False
    return True

@cuda.jit(device=True)
def board_equals_gpu(b1, b2):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if b1[r,c] != b2[r,c]:
                return False
    return True

@cuda.jit(device=True)
def move_board_gpu(board_in, move_id):
    local_in = cuda.local.array((GRID_SIZE, GRID_SIZE), dtype=int32)
    local_out = cuda.local.array((GRID_SIZE, GRID_SIZE), dtype=int32)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            local_in[r,c] = board_in[r,c]

    for rr in range(GRID_SIZE):
        for cc in range(GRID_SIZE):
            local_out[rr,cc] = 0

    score_acc = 0

    # LEFT
    if move_id == 2:
        for rr in range(GRID_SIZE):
            row_non_zero = cuda.local.array(GRID_SIZE, int32)
            nz_count = 0
            for cc in range(GRID_SIZE):
                val = local_in[rr, cc]
                if val != 0:
                    row_non_zero[nz_count] = val
                    nz_count += 1
            merged = cuda.local.array(GRID_SIZE, int32)
            for i in range(GRID_SIZE):
                merged[i] = 0
            m_i = 0
            i = 0
            while i < nz_count:
                val = row_non_zero[i]
                if i < nz_count-1 and val == row_non_zero[i+1]:
                    val *= 2
                    score_acc += val
                    merged[m_i] = val
                    m_i += 1
                    i += 2
                else:
                    merged[m_i] = val
                    m_i += 1
                    i += 1
            for cc in range(GRID_SIZE):
                if cc < m_i:
                    local_out[rr, cc] = merged[cc]
                else:
                    local_out[rr, cc] = 0

    # RIGHT
    elif move_id == 3:
        for rr in range(GRID_SIZE):
            row_non_zero = cuda.local.array(GRID_SIZE, int32)
            nz_count = 0
            for cc in range(GRID_SIZE-1, -1, -1):
                val = local_in[rr, cc]
                if val != 0:
                    row_non_zero[nz_count] = val
                    nz_count += 1
            merged = cuda.local.array(GRID_SIZE, int32)
            for i in range(GRID_SIZE):
                merged[i] = 0
            m_i = 0
            i = 0
            while i < nz_count:
                val = row_non_zero[i]
                if i < nz_count-1 and val == row_non_zero[i+1]:
                    val *= 2
                    score_acc += val
                    merged[m_i] = val
                    m_i += 1
                    i += 2
                else:
                    merged[m_i] = val
                    m_i += 1
                    i += 1
            for cc in range(GRID_SIZE):
                if cc < m_i:
                    local_out[rr, GRID_SIZE-1-cc] = merged[cc]
                else:
                    local_out[rr, GRID_SIZE-1-cc] = 0

    # UP
    elif move_id == 0:
        for cc in range(GRID_SIZE):
            col_non_zero = cuda.local.array(GRID_SIZE, int32)
            nz_count = 0
            for rr in range(GRID_SIZE):
                val = local_in[rr, cc]
                if val != 0:
                    col_non_zero[nz_count] = val
                    nz_count += 1
            merged = cuda.local.array(GRID_SIZE, int32)
            for i in range(GRID_SIZE):
                merged[i] = 0
            m_i = 0
            i = 0
            while i < nz_count:
                val = col_non_zero[i]
                if i < nz_count-1 and val == col_non_zero[i+1]:
                    val *= 2
                    score_acc += val
                    merged[m_i] = val
                    m_i += 1
                    i += 2
                else:
                    merged[m_i] = val
                    m_i += 1
                    i += 1
            for rr2 in range(GRID_SIZE):
                if rr2 < m_i:
                    local_out[rr2, cc] = merged[rr2]
                else:
                    local_out[rr2, cc] = 0

    # DOWN
    else:
        for cc in range(GRID_SIZE):
            col_non_zero = cuda.local.array(GRID_SIZE, int32)
            nz_count = 0
            for rr in range(GRID_SIZE-1, -1, -1):
                val = local_in[rr, cc]
                if val != 0:
                    col_non_zero[nz_count] = val
                    nz_count += 1
            merged = cuda.local.array(GRID_SIZE, int32)
            for i in range(GRID_SIZE):
                merged[i] = 0
            m_i = 0
            i = 0
            while i < nz_count:
                val = col_non_zero[i]
                if i < nz_count-1 and val == col_non_zero[i+1]:
                    val *= 2
                    score_acc += val
                    merged[m_i] = val
                    m_i += 1
                    i += 2
                else:
                    merged[m_i] = val
                    m_i += 1
                    i += 1
            for rr2 in range(GRID_SIZE):
                if rr2 < m_i:
                    local_out[GRID_SIZE-1-rr2, cc] = merged[rr2]
                else:
                    local_out[GRID_SIZE-1-rr2, cc] = 0

    return local_out, score_acc

@cuda.jit
def random_rollout_kernel(boards_in, results_out, max_moves, rng_states):
    i = cuda.grid(1)
    if i >= boards_in.shape[0]:
        return

    board_copy = cuda.local.array((GRID_SIZE, GRID_SIZE), dtype=int32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            board_copy[r, c] = boards_in[i, r, c]

    seed = rng_states[i]
    def lcg_rand(s):
        return (s * 1664525 + 1013904223) & 0xFFFFFFFF

    moves_count = 0
    while moves_count < max_moves:
        # test cases vides
        found_zero = False
        for rr in range(GRID_SIZE):
            for cc in range(GRID_SIZE):
                if board_copy[rr, cc] == 0:
                    found_zero = True
                    break
            if found_zero:
                break

        if not found_zero:
            can_move = False
            for rr in range(GRID_SIZE):
                for cc in range(GRID_SIZE-1):
                    if board_copy[rr, cc] == board_copy[rr, cc+1]:
                        can_move = True
                        break
                if can_move:
                    break
            for rr in range(GRID_SIZE-1):
                for cc in range(GRID_SIZE):
                    if board_copy[rr, cc] == board_copy[rr+1, cc]:
                        can_move = True
                        break
                if can_move:
                    break
            if not can_move:
                break

        possible_moves = cuda.local.array(4, int32)
        pm_count = 0
        for mid in range(4):
            new_b, _ = move_board_gpu(board_copy, mid)
            eq = board_equals_gpu(new_b, board_copy)
            if not eq:
                possible_moves[pm_count] = mid
                pm_count += 1

        if pm_count == 0:
            break

        seed = lcg_rand(seed)
        idx = seed % pm_count
        move_id = possible_moves[idx]

        new_b, _ = move_board_gpu(board_copy, move_id)
        for rr in range(GRID_SIZE):
            for cc in range(GRID_SIZE):
                board_copy[rr, cc] = new_b[rr, cc]

        empties = cuda.local.array((16,2), int32)
        ecount = 0
        for rr in range(GRID_SIZE):
            for cc in range(GRID_SIZE):
                if board_copy[rr, cc] == 0:
                    empties[ecount, 0] = rr
                    empties[ecount, 1] = cc
                    ecount += 1
        if ecount > 0:
            seed = lcg_rand(seed)
            e_idx = seed % ecount
            er = empties[e_idx, 0]
            ec = empties[e_idx, 1]
            seed = lcg_rand(seed)
            if (seed & 0xFF) < 230:
                tile_val = 2
            else:
                tile_val = 4
            board_copy[er, ec] = tile_val

        moves_count += 1

    res_score = 0
    for rr in range(GRID_SIZE):
        for cc in range(GRID_SIZE):
            res_score += board_copy[rr, cc]

    results_out[i] = res_score


def evaluate_move_gpu(board, move_id, rollouts=8192):
    new_b, _ = apply_move_njit(board, move_id)
    if board_equals(new_b, board):
        return -1.0

    new_b = new_b.copy()
    new_b = spawn_new_tile(new_b)

    boards_batch = np.stack([new_b]*rollouts, axis=0)  # (rollouts,4,4)
    final_scores = np.zeros(rollouts, dtype=np.int32)

    rng_host = np.array([random.randint(1,2**31-1) for _ in range(rollouts)], dtype=np.uint32)

    num_streams = 8
    chunk_size = rollouts // num_streams

    streams = [cuda.stream() for _ in range(num_streams)]

    d_boards_sub = []
    d_results_sub = []
    d_rng_sub = []

    threads_per_block = 16
    index_offset = 0
    for s in range(num_streams):
        start_idx = s * chunk_size
        end_idx = start_idx + chunk_size
        if s == num_streams - 1:
            end_idx = rollouts
        sub_size = end_idx - start_idx

        boards_sub = boards_batch[start_idx:end_idx]
        rng_sub = rng_host[start_idx:end_idx]

        d_b = cuda.to_device(boards_sub, stream=streams[s])
        # ICI on utilise np.int32, pas int32
        d_r = cuda.device_array(sub_size, dtype=np.int32, stream=streams[s])
        d_rng = cuda.to_device(rng_sub, stream=streams[s])

        d_boards_sub.append(d_b)
        d_results_sub.append(d_r)
        d_rng_sub.append(d_rng)

        blocks = (sub_size + threads_per_block - 1) // threads_per_block
        random_rollout_kernel[blocks, threads_per_block, streams[s]](d_b, d_r, 200, d_rng)

    index_offset = 0
    for s in range(num_streams):
        streams[s].synchronize()
        sub_result = d_results_sub[s].copy_to_host(stream=streams[s])
        sub_size = sub_result.size
        final_scores[index_offset:index_offset+sub_size] = sub_result
        index_offset += sub_size

    return np.mean(final_scores)

def choose_best_move_gpu(board, rollouts=8192):
    moves = get_possible_moves(board)
    if not moves:
        return None
    best_score = -1
    best_move = None
    for m in moves:
        sc = evaluate_move_gpu(board, m, rollouts)
        if sc > best_score:
            best_score = sc
            best_move = m
    return best_move

def init_board():
    b = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    b = spawn_new_tile(b)
    b = spawn_new_tile(b)
    return b

def play_2048_gpu(n_steps=50, rollouts=8192):
    board = init_board()
    start_time = time.time()

    for step in range(n_steps):
        if is_game_over(board):
            break
        move = choose_best_move_gpu(board, rollouts)
        if move is None:
            break
        new_b, _ = apply_move_njit(board, move)
        board = new_b.copy()
        board = spawn_new_tile(board)

        # === Affichage live de l'avancement ===
        elapsed = time.time() - start_time
        current_score = get_score_njit(board)
        possible_moves = get_possible_moves(board)
        move_str = ["UP", "DOWN", "LEFT", "RIGHT"][move]
        print(f"[{step+1}/{n_steps}] Move: {move_str}, Score: {current_score}, "
              f"Moves left: {[['UP', 'DOWN', 'LEFT', 'RIGHT'][m] for m in possible_moves]}, "
              f"Time: {elapsed:.2f}s")

    return board

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=== 2048 GPU ===")
    start_t = time.time()

    final_board = play_2048_gpu(n_steps=10000, rollouts=64)

    end_t = time.time()

    print("Partie terminée !")
    print(final_board)
    print(f"Score final = {get_score_njit(final_board)}")
    print(f"Temps écoulé : {end_t - start_t:.2f} s")
