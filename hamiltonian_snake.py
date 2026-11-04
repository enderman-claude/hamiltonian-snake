#!/usr/bin/env python3
"""
Hamiltonian Snake with a search-based AI.

Runs in a pygame window when pygame and a display are available. Otherwise it
runs headless: the AI plays in the terminal and reports results.

  python hamiltonian_snake.py                       # window if possible, else headless
  python hamiltonian_snake.py --headless --games 20 --seed 1
  python hamiltonian_snake.py --headless --load hamiltonian_save.json --show

Window keys: SPACE pause · TAB human/AI · arrows or WASD move (human)
             N new game · C new cycle · G show cycle · F5 save · F9 load
             +/- speed · ESC quit
"""
import argparse
import json
import os
import random
import sys
import time
from collections import deque

W = H = 20
N = W * H
CELL = 26
SAVE_FILE = "hamiltonian_save.json"
DIR_LIST = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

REWARD_FOOD = 10
REWARD_CYCLE_BASE = 5
REWARD_CYCLE_STREAK_CAP = 20
REWARD_CLOSE = 1000        # body covers all cells and closes into a Hamiltonian cycle
SHORTCUT_BONUS = 4         # AI value for an off-cycle step that gets closer to food
DANGER_PENALTY = 300       # AI value penalty when the tail becomes unreachable
DISCOUNT = 0.9             # AI discount for later moves in the search
HUNGER_LIMIT = 300         # moves without food before the snake starves (stops endless cycling)

# The AI plans for growth and loop completion, not for raw points. If it planned
# on raw points, circling the cycle would pay more than eating and it would never eat.
AI_EAT = 100.0
AI_CYCLE_STEP = 0.5


# ---------------------------------------------------------------- geometry
def idx(x, y):
    return y * W + x


def xy(i):
    return i % W, i // W


def inb(x, y):
    return 0 <= x < W and 0 <= y < H


def adjacent(a, b):
    ax, ay = xy(a)
    bx, by = xy(b)
    return abs(ax - bx) + abs(ay - by) == 1


def opposite(d):
    return (-d[0], -d[1])


# ---------------------------------------------------------------- Hamiltonian cycles
def comb_cycle():
    """Zigzag Hamiltonian cycle on the W x H grid (W and H both even)."""
    o = [idx(x, 0) for x in range(W)]
    for y in range(1, H):
        xs = range(W - 1, 0, -1) if y % 2 else range(1, W)
        o.extend(idx(x, y) for x in xs)
    o.extend(idx(0, y) for y in range(H - 1, 0, -1))
    return o


def order_from_adj(A):
    """Walk a 2-regular adjacency list from cell 0 and return the cells in loop order."""
    o, prev, cur = [0], -1, 0
    for _ in range(N - 1):
        a, b = A[cur]
        nxt = a if a != prev else b
        prev, cur = cur, nxt
        o.append(cur)
    return o


def random_hamiltonian_cycle(rng):
    """Random Hamiltonian cycle from a random spanning tree of 2x2 blocks.

    Every block of four cells starts as a small ring. For each tree edge between
    neighbouring blocks, one side of each block's ring is cut and the two cut
    ends are reconnected across the boundary. Each cell keeps exactly two
    neighbours, and the result is a single loop through all cells. Different
    trees give different loops, so the shape varies from game to game."""
    bw, bh = W // 2, H // 2
    A = [[] for _ in range(N)]

    def block_cells(i, j):                    # clockwise: TL, TR, BR, BL
        return [idx(2 * i, 2 * j), idx(2 * i + 1, 2 * j),
                idx(2 * i + 1, 2 * j + 1), idx(2 * i, 2 * j + 1)]

    for i in range(bw):
        for j in range(bh):
            cs = block_cells(i, j)
            for k in range(4):
                a, b = cs[k], cs[(k + 1) % 4]
                A[a].append(b)
                A[b].append(a)

    def swap(c, old, new):
        A[c][A[c].index(old)] = new

    # random spanning tree over blocks (randomised Kruskal)
    parent = list(range(bw * bh))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    edges = []
    for i in range(bw):
        for j in range(bh):
            if i + 1 < bw:
                edges.append(((i, j), (i + 1, j), 'h'))
            if j + 1 < bh:
                edges.append(((i, j), (i, j + 1), 'v'))
    rng.shuffle(edges)

    for (i1, j1), (i2, j2), kind in edges:
        u, v = i1 * bh + j1, i2 * bh + j2
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        parent[ru] = rv
        if kind == 'h':                       # block (i,j) -> block (i+1,j)
            TR1, BR1 = idx(2 * i1 + 1, 2 * j1), idx(2 * i1 + 1, 2 * j1 + 1)
            TL2, BL2 = idx(2 * i2, 2 * j2), idx(2 * i2, 2 * j2 + 1)
            swap(TR1, BR1, TL2); swap(BR1, TR1, BL2)
            swap(TL2, BL2, TR1); swap(BL2, TL2, BR1)
        else:                                 # block (i,j) -> block (i,j+1)
            BL1, BR1 = idx(2 * i1, 2 * j1 + 1), idx(2 * i1 + 1, 2 * j1 + 1)
            TL2, TR2 = idx(2 * i2, 2 * j2), idx(2 * i2 + 1, 2 * j2)
            swap(BL1, BR1, TL2); swap(BR1, BL1, TR2)
            swap(TL2, TR2, BL1); swap(TR2, TL2, BR1)

    return order_from_adj(A)


# ---------------------------------------------------------------- rules
def simulate(snake, food, d, cadj):
    """Pure one-step move. Returns (new_snake, ate, on_cycle) or None if fatal."""
    h = snake[0]
    x, y = xy(h)
    nx, ny = x + d[0], y + d[1]
    if not inb(nx, ny):
        return None
    nh = idx(nx, ny)
    eat = nh == food
    body = snake if eat else snake[:-1]   # the tail moves away unless we eat
    if nh in body:
        return None
    return [nh] + body, eat, nh in cadj[h]


def flood(start, blocked, target=None):
    """BFS over free cells. Returns (distance to target or -1, reachable cell count)."""
    seen = bytearray(N)
    seen[start] = 1
    q = deque([(start, 0)])
    count, found = 0, -1
    while q:
        c, d = q.popleft()
        count += 1
        if c == target and found < 0:
            found = d
        x, y = xy(c)
        for dx, dy in DIR_LIST:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                n = idx(nx, ny)
                if not seen[n] and n not in blocked:
                    seen[n] = 1
                    q.append((n, d + 1))
    return found, count


class Game:
    def __init__(self, order, rng=None):
        self.rng = rng or random.Random()
        self.set_cycle(order)
        self.reset()

    def set_cycle(self, order):
        self.order = list(order)
        self.cadj = [None] * N
        for i, c in enumerate(self.order):
            self.cadj[c] = (self.order[i - 1], self.order[(i + 1) % N])

    def reset(self):
        k = self.rng.randrange(3, N)
        o = self.order
        self.snake = [o[k], o[k - 1], o[k - 2]]
        hx, hy = xy(o[k])
        bx, by = xy(o[k - 1])
        self.dir = (hx - bx, hy - by)
        self.pending = None
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.cycle_steps = 0
        self.shortcuts = 0
        self.since_food = 0
        self.over = False
        self.won = False
        self.hamiltonian = False
        self.food = None
        self.place_food()
        self.message = "New game."

    def place_food(self):
        occ = set(self.snake)
        free = [c for c in range(N) if c not in occ]
        self.food = self.rng.choice(free)

    def move(self, d, shortcut=False):
        if self.over:
            return
        r = simulate(self.snake, self.food, d, self.cadj)
        if r is None:
            self._end(f"Crashed. Score {self.score}.", False)
            return
        new, eat, on = r
        self.snake = new
        self.dir = d
        if on:
            self.streak += 1
            self.cycle_steps += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.score += REWARD_CYCLE_BASE + min(self.streak, REWARD_CYCLE_STREAK_CAP)
        else:
            self.streak = 0
        if shortcut:
            self.shortcuts += 1
        self.since_food = 0 if eat else self.since_food + 1
        if eat:
            self.score += REWARD_FOOD
            if len(self.snake) == N:
                closed = adjacent(self.snake[0], self.snake[-1])
                if closed:
                    self.score += REWARD_CLOSE
                    self.hamiltonian = True
                    self._end(f"Hamiltonian cycle closed! Score {self.score}.", True)
                else:
                    self._end(f"Board filled, loop not closed. Score {self.score}.", True)
                return
            self.place_food()
        elif self.since_food > HUNGER_LIMIT:
            self._end(f"Starved after {HUNGER_LIMIT} moves without food. Score {self.score}.", False)

    def queue(self, d):
        """Human input: ignore reversals."""
        if d != opposite(self.dir):
            self.pending = d

    def _end(self, msg, won):
        self.over = True
        self.won = won
        self.message = msg

    # ---- save / load
    def to_dict(self):
        return {
            "v": 1, "order": self.order, "snake": self.snake, "dir": list(self.dir),
            "food": self.food, "score": self.score, "streak": self.streak,
            "best_streak": self.best_streak, "cycle_steps": self.cycle_steps,
            "shortcuts": self.shortcuts, "since_food": self.since_food,
            "over": self.over, "won": self.won,
            "hamiltonian": self.hamiltonian,
        }

    @classmethod
    def from_dict(cls, d, rng=None):
        def cell(c):
            return isinstance(c, int) and not isinstance(c, bool) and 0 <= c < N

        order = d.get("order")
        snake = d.get("snake")
        if d.get("v") != 1:
            raise ValueError("unknown save version")
        if not (isinstance(order, list) and len(order) == N and set(order) == set(range(N))):
            raise ValueError("cycle in save is not a Hamiltonian cycle of the board")
        if not (isinstance(snake, list) and len(snake) >= 2 and all(cell(c) for c in snake)
                and len(set(snake)) == len(snake)):
            raise ValueError("snake in save is invalid")
        if not d.get("over") and not cell(d.get("food")):
            raise ValueError("food position in save is invalid")

        g = cls.__new__(cls)
        g.rng = rng or random.Random()
        g.set_cycle(order)
        g.snake = snake
        g.dir = tuple(d.get("dir", (1, 0)))
        g.pending = None
        g.food = d.get("food")
        g.score = int(d.get("score", 0))
        g.streak = int(d.get("streak", 0))
        g.best_streak = int(d.get("best_streak", 0))
        g.cycle_steps = int(d.get("cycle_steps", 0))
        g.shortcuts = int(d.get("shortcuts", 0))
        g.since_food = int(d.get("since_food", 0))
        g.over = bool(d.get("over", False))
        g.won = bool(d.get("won", False))
        g.hamiltonian = bool(d.get("hamiltonian", False))
        g.message = "Loaded."
        return g


def save_game(game, path=SAVE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(game.to_dict(), f)


def load_game(path=SAVE_FILE, rng=None):
    with open(path, encoding="utf-8") as f:
        return Game.from_dict(json.load(f), rng)


# ---------------------------------------------------------------- AI
class SearchAI:
    """Look-ahead planner.

    It only considers moves after which the tail is still reachable, so it never
    traps itself. Each candidate is scored by searching `depth` moves ahead and
    summing the game's own points plus a heuristic: food proximity, open space,
    and a bonus for shortcuts (off-cycle steps that close the distance to food).
    It is a search with a hand-tuned evaluation, not a learned policy.
    """

    def __init__(self, depth=2):
        self.depth = depth
        self.last_shortcut = False

    def choose(self, game):
        self.last_shortcut = False
        head = game.snake[0]
        d0 = flood(head, set(game.snake[:-1]), game.food)[0] if game.food is not None else -1
        best_d, best_v, best_short = None, float("-inf"), False

        for d in DIR_LIST:
            if d == opposite(game.dir):
                continue
            r = simulate(game.snake, game.food, d, game.cadj)
            if r is None:
                continue
            new, eat, on = r
            v, short = self._step_reward(game.snake, new, eat, on, game.streak, d0, game.food)
            nf = None if eat else game.food
            v += DISCOUNT * self._search(new, nf, game.cadj, self._streak_after(game.streak, on),
                                         self.depth - 1, d)
            if v > best_v:
                best_d, best_v, best_short = d, v, short

        if best_d is None:
            return game.dir          # no legal move; the game will report the crash
        self.last_shortcut = best_short
        return best_d

    @staticmethod
    def _streak_after(streak, on):
        return streak + 1 if on else 0

    def _step_reward(self, old, new, eat, on, streak, d0, food):
        v = 0.0
        if on:
            v += AI_CYCLE_STEP
        if eat:
            v += AI_EAT
        shortcut = False
        if food is not None and not on and not eat and d0 >= 0:
            d1 = flood(new[0], set(new[:-1]), food)[0]
            if 0 <= d1 < d0:
                v += SHORTCUT_BONUS
                shortcut = True
        if flood(new[0], set(new[:-1]), new[-1])[0] < 0:
            v -= DANGER_PENALTY
        return v, shortcut

    def _search(self, snake, food, cadj, streak, depth, prev_d):
        if depth == 0:
            return self._evaluate(snake, food)
        d0 = flood(snake[0], set(snake[:-1]), food)[0] if food is not None else -1
        best, moved = float("-inf"), False
        for d in DIR_LIST:
            if d == opposite(prev_d):
                continue
            r = simulate(snake, food, d, cadj)
            if r is None:
                continue
            moved = True
            new, eat, on = r
            v, _ = self._step_reward(snake, new, eat, on, streak, d0, food)
            nf = None if eat else food
            v += DISCOUNT * self._search(new, nf, cadj, self._streak_after(streak, on), depth - 1, d)
            best = max(best, v)
        return best if moved else -1000.0

    def _evaluate(self, snake, food):
        head, blocked = snake[0], set(snake[:-1])
        tail_d, space = flood(head, blocked, snake[-1])
        if tail_d < 0:
            return -DANGER_PENALTY
        v = 0.1 * space
        if food is not None:
            fd = flood(head, blocked, food)[0]
            v -= 2.0 * fd if fd >= 0 else 60.0
        return v


# ---------------------------------------------------------------- graphics detection
def detect_pygame(force_window=False):
    """Return the pygame module if a window can be opened, otherwise None."""
    try:
        import pygame
    except ImportError:
        print("pygame is not installed; running headless.", file=sys.stderr)
        return None
    try:
        pygame.display.init()
        if not force_window and pygame.display.get_driver() == "dummy":
            pygame.display.quit()
            print("No display available; running headless.", file=sys.stderr)
            return None
        return pygame
    except pygame.error as e:
        print(f"No display available ({e}); running headless.", file=sys.stderr)
        return None


# ---------------------------------------------------------------- headless
def ascii_board(game):
    grid = [["." for _ in range(W)] for _ in range(H)]
    if game.food is not None:
        fx, fy = xy(game.food)
        grid[fy][fx] = "*"
    for i, c in enumerate(game.snake):
        x, y = xy(c)
        grid[y][x] = "@" if i == 0 else "o"
    return "\n".join("".join(row) for row in grid)


def run_headless(args, rng):
    """Play AI games until --games is reached, or until --minutes has elapsed.
    Each finished game is appended to --out as one JSON line."""
    ai = SearchAI(depth=args.depth)
    limit = N * N * 2
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    results, last = [], None
    t_start = time.time()

    loaded = load_game(args.load, rng) if args.load else None
    i = 0
    while True:
        if deadline is not None:
            if time.time() >= deadline:
                break
        elif i >= args.games:
            break
        if loaded is not None:
            game, loaded = loaded, None
        else:
            game = Game(random_hamiltonian_cycle(rng), rng)

        t0 = time.time()
        steps = 0
        while not game.over and steps < limit:
            if deadline is not None and time.time() >= deadline and steps % 50 == 0:
                break   # time budget spent mid-game; record what we have
            game.move(ai.choose(game), shortcut=ai.last_shortcut)
            steps += 1

        record = {
            "game": i + 1, "score": game.score, "length": len(game.snake), "cells": N,
            "cycle_steps": game.cycle_steps, "shortcuts": game.shortcuts,
            "hamiltonian": game.hamiltonian, "won": game.won, "finished": game.over,
            "steps": steps, "seconds": round(time.time() - t0, 2), "message": game.message,
        }
        results.append(record)
        last = game
        if out:
            out.write(json.dumps(record) + "\n")
            out.flush()
        print(f"game {i + 1:3d}: score {game.score:7d}  length {len(game.snake):3d}/{N}  "
              f"cycle steps {game.cycle_steps:6d}  shortcuts {game.shortcuts:5d}  "
              f"Hamiltonian {'yes' if game.hamiltonian else 'no '}  "
              f"{'won' if game.won else 'lost'}", flush=True)
        i += 1

    if out:
        out.close()
    elapsed = time.time() - t_start
    if results:
        scores = [r["score"] for r in results]
        print("-" * 78)
        print(f"games {len(results)}  mean score {sum(scores) / len(scores):.1f}  "
              f"best {max(scores)}  Hamiltonian loops {sum(r['hamiltonian'] for r in results)}  "
              f"time {elapsed:.1f}s")
    if args.show and last is not None:
        print(last.message)
        print(ascii_board(last))
    if args.save and last is not None:
        save_game(last, args.save)
        print(f"saved to {args.save}")


# ---------------------------------------------------------------- window
PALETTE = {
    "paper": (243, 239, 228), "board": (235, 230, 215), "tile": (228, 222, 206),
    "chalk": (195, 184, 156), "snake": (61, 108, 147), "head": (31, 42, 68),
    "food": (217, 87, 59), "ink": (31, 42, 68), "muted": (109, 106, 95),
}
PANEL_W = 290
KEYMAP = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


def run_window(pg, args, rng):
    pg.init()
    pg.display.set_caption("Hamiltonian Snake")
    screen = pg.display.set_mode((W * CELL + PANEL_W, H * CELL))
    font = pg.font.SysFont(None, 26)
    big = pg.font.SysFont(None, 40)
    clock = pg.time.Clock()

    game = Game(random_hamiltonian_cycle(rng), rng)
    if args.load:
        try:
            game = load_game(args.load, rng)
        except (OSError, ValueError) as e:
            game.message = f"Could not load: {e}"
    ai = SearchAI(depth=args.depth)
    mode = "ai"
    paused = False
    show_cycle = False
    delay = 90
    acc = 0

    key_dirs = {
        pg.K_UP: (0, -1), pg.K_w: (0, -1), pg.K_DOWN: (0, 1), pg.K_s: (0, 1),
        pg.K_LEFT: (-1, 0), pg.K_a: (-1, 0), pg.K_RIGHT: (1, 0), pg.K_d: (1, 0),
    }
    running = True
    while running:
        dt = clock.tick(60)
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                k = ev.key
                if k == pg.K_ESCAPE:
                    running = False
                elif k == pg.K_SPACE:
                    paused = not paused
                elif k == pg.K_TAB:
                    mode = "human" if mode == "ai" else "ai"
                    game.pending = None
                    game.message = "Human control." if mode == "human" else "AI control."
                elif k == pg.K_n:
                    game.reset()
                elif k == pg.K_c:
                    game.set_cycle(random_hamiltonian_cycle(rng))
                    game.reset()
                    game.message = "New Hamiltonian cycle."
                elif k == pg.K_g:
                    show_cycle = not show_cycle
                elif k == pg.K_F5:
                    try:
                        save_game(game)
                        game.message = f"Saved to {SAVE_FILE}."
                    except OSError as e:
                        game.message = f"Save failed: {e}"
                elif k == pg.K_F9:
                    try:
                        game = load_game(SAVE_FILE, rng)
                        paused = True
                        game.message = "Loaded. Press SPACE to continue."
                    except (OSError, ValueError) as e:
                        game.message = f"Load failed: {e}"
                elif k in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
                    delay = max(20, delay - 10)
                elif k in (pg.K_MINUS, pg.K_KP_MINUS):
                    delay = min(400, delay + 10)
                elif k in key_dirs and mode == "human":
                    game.queue(key_dirs[k])

        if not paused and not game.over:
            acc += dt
            if acc >= delay:
                acc = 0
                if mode == "ai":
                    game.move(ai.choose(game), shortcut=ai.last_shortcut)
                else:
                    d = game.pending or game.dir
                    game.pending = None
                    game.move(d)
        draw(pg, screen, font, big, game, show_cycle, mode, delay, paused, ai.depth)
        pg.display.flip()
        if args.frames and clock.get_fps() >= 0:
            args.frames -= 1
            if args.frames <= 0:
                running = False
    pg.quit()


def draw(pg, screen, font, big, game, show_cycle, mode, delay, paused, depth):
    C = PALETTE
    screen.fill(C["paper"])
    pg.draw.rect(screen, C["board"], (0, 0, W * CELL, H * CELL))
    for y in range(H):
        for x in range(W):
            if (x + y) % 2:
                pg.draw.rect(screen, C["tile"], (x * CELL, y * CELL, CELL, CELL))
    if show_cycle:
        pts = []
        for c in game.order:
            x, y = xy(c)
            pts.append((x * CELL + CELL // 2, y * CELL + CELL // 2))
        pg.draw.lines(screen, C["chalk"], True, pts, 3)
    if game.food is not None:
        fx, fy = xy(game.food)
        pg.draw.circle(screen, C["food"], (fx * CELL + CELL // 2, fy * CELL + CELL // 2), CELL // 3)
    for i, c in enumerate(game.snake):
        x, y = xy(c)
        pg.draw.rect(screen, C["head"] if i == 0 else C["snake"],
                     (x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4), border_radius=6)

    px = W * CELL + 18
    rows = [
        ("Score", f"{game.score}"), ("Length", f"{len(game.snake)}/{N}"),
        ("Cycle streak", f"{game.streak}"), ("Cycle steps", f"{game.cycle_steps}"),
        ("Shortcuts", f"{game.shortcuts}"), ("Mode", f"{'AI, depth ' + str(depth) if mode == 'ai' else 'Human'}"),
        ("Delay", f"{delay} ms"),
    ]
    y = 14
    for label, value in rows:
        screen.blit(font.render(label, True, C["muted"]), (px, y))
        screen.blit(big.render(value, True, C["ink"]), (px + 120, y - 6))
        y += 36
    y += 8
    status = "Paused" if paused else ("Game over" if game.over else "Running")
    screen.blit(big.render(status, True, C["ink"]), (px, y))
    y += 44
    for line in _wrap(game.message, 34):
        screen.blit(font.render(line, True, C["ink"]), (px, y))
        y += 22
    y += 18
    help_lines = [
        "SPACE pause   TAB mode   N new game",
        "C new cycle   G show cycle",
        "F5 save   F9 load   +/- speed   ESC quit",
    ]
    for line in help_lines:
        screen.blit(font.render(line, True, C["muted"]), (px, y))
        y += 22


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------- entry point
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="never open a window")
    ap.add_argument("--window", action="store_true", help="open a window even with a dummy video driver")
    ap.add_argument("--games", type=int, default=3, help="headless: number of games to play")
    ap.add_argument("--depth", type=int, default=2, help="AI look-ahead depth in moves")
    ap.add_argument("--seed", type=int, help="random seed for reproducible runs")
    ap.add_argument("--load", metavar="FILE", help="resume a saved game")
    ap.add_argument("--save", metavar="FILE", help="headless: save the last game")
    ap.add_argument("--show", action="store_true", help="headless: print the final board")
    ap.add_argument("--minutes", type=float, default=0, help="headless: play until this many minutes have passed")
    ap.add_argument("--out", metavar="FILE", help="headless: append one JSON line per game")
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pg = None if args.headless else detect_pygame(force_window=args.window)
    if pg is None:
        run_headless(args, rng)
    else:
        run_window(pg, args, rng)


if __name__ == "__main__":
    main()
