import random
import unittest

from hamiltonian_snake import (N, Game, SearchAI, adjacent, comb_cycle,
                               random_hamiltonian_cycle, simulate, idx)


class CycleTests(unittest.TestCase):
    def assert_hamiltonian(self, order):
        self.assertEqual(len(order), N)
        self.assertEqual(set(order), set(range(N)))
        for i in range(N):
            self.assertTrue(adjacent(order[i], order[(i + 1) % N]))

    def test_comb_cycle_is_hamiltonian(self):
        self.assert_hamiltonian(comb_cycle())

    def test_random_cycles_are_hamiltonian_and_differ(self):
        seen = set()
        for seed in range(5):
            order = random_hamiltonian_cycle(random.Random(seed))
            self.assert_hamiltonian(order)
            seen.add(tuple(order))
        self.assertGreater(len(seen), 1)


class RuleTests(unittest.TestCase):
    def test_wall_and_self_collision_are_fatal(self):
        cadj = [None] * N
        self.assertIsNone(simulate([idx(0, 0), idx(1, 0), idx(2, 0)], 5, (0, -1), cadj))
        # head at (5,5) moving up into (5,4), which is the 4th body cell here
        loop = [idx(5, 5), idx(5, 4), idx(4, 4), idx(4, 5), idx(4, 6)]
        self.assertIsNone(simulate(loop, 0, (0, -1), cadj))

    def test_save_round_trip(self):
        rng = random.Random(3)
        g = Game(random_hamiltonian_cycle(rng), rng)
        ai = SearchAI(depth=1)
        for _ in range(30):
            g.move(ai.choose(g), shortcut=ai.last_shortcut)
        g2 = Game.from_dict(g.to_dict())
        self.assertEqual(g2.snake, g.snake)
        self.assertEqual(g2.food, g.food)
        self.assertEqual(g2.score, g.score)

    def test_invalid_save_is_rejected(self):
        rng = random.Random(4)
        d = Game(random_hamiltonian_cycle(rng), rng).to_dict()
        d["order"] = list(range(N - 1)) + [0]
        with self.assertRaises(ValueError):
            Game.from_dict(d)


class AITests(unittest.TestCase):
    def test_ai_plays_a_game_without_crashing_early(self):
        rng = random.Random(11)
        g = Game(random_hamiltonian_cycle(rng), rng)
        ai = SearchAI(depth=1)
        for _ in range(200):
            if g.over:
                break
            g.move(ai.choose(g), shortcut=ai.last_shortcut)
        self.assertFalse(g.over, g.message)


if __name__ == "__main__":
    unittest.main()
