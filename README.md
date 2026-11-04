# Hamiltonian Snake

Snake on a 20×20 board where the snake's path follows a Hamiltonian cycle.
A search-based AI plays it; the game can also be played by hand in a pygame window.

## Run

```bash
python hamiltonian_snake.py                       # window if pygame and a display are available
python hamiltonian_snake.py --headless --games 5  # terminal only
python hamiltonian_snake.py --headless --minutes 60 --out results.jsonl
```

Windowed play needs `pip install pygame`. Headless runs use the standard library only.

## Scoring

- Food: 10 points.
- Each step along an edge of the Hamiltonian cycle: 5 points plus a streak bonus (up to 20).
- Closing a loop through all cells: 1000-point bonus.

The AI plans for growth and loop completion rather than raw points.

## Tests

```bash
python -m unittest discover -s tests -v
```

## CI and timed runs

- `ci.yml` runs the tests and a one-minute smoke run on every push and pull request.
- `timed-runs.yml` is started manually (Actions → timed-runs → Run workflow). It plays for
  1h, 2h, 4h, 8h, 1d, 2d, 4d, 8d and 30d and uploads the results as artifacts.
  The 8h and longer rows need a self-hosted runner with the label `long-run`.
