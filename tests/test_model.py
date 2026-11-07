import unittest

from kvlens.model import Cache, Model, cache_token, forward_all, generate_cached, generate_full


class ModelInitTests(unittest.TestCase):
    def test_shapes(self):
        m = Model(vocab=10, dim=8, heads=2, seed=0)
        self.assertEqual(len(m.embed), 10)
        self.assertEqual(len(m.wq), 2)
        self.assertEqual(len(m.wout[0]), 10)
        self.assertEqual(m.head_dim, 4)

    def test_rejects_bad_heads(self):
        with self.assertRaises(ValueError):
            Model(vocab=10, dim=7, heads=2)
        with self.assertRaises(ValueError):
            Model(vocab=10, dim=8, heads=0)

    def test_rejects_tiny_vocab(self):
        with self.assertRaises(ValueError):
            Model(vocab=1, dim=8, heads=2)

    def test_seeded_determinism(self):
        a = Model(vocab=6, dim=4, heads=2, seed=9)
        b = Model(vocab=6, dim=4, heads=2, seed=9)
        self.assertEqual(a.wq, b.wq)


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.m = Model(vocab=10, dim=8, heads=2, seed=2)

    def test_logits_shape(self):
        logits = forward_all(self.m, [1, 2, 3])
        self.assertEqual(len(logits), 10)

    def test_deterministic(self):
        self.assertEqual(forward_all(self.m, [1, 2, 3]), forward_all(self.m, [1, 2, 3]))

    def test_cache_grows(self):
        c = Cache(self.m)
        for t in [0, 1, 2]:
            cache_token(self.m, c, t)
        self.assertEqual(c.length, 3)
        self.assertEqual(len(c.k[0]), 3)

    def test_counter_increments(self):
        ctr = {"flops": 0}
        forward_all(self.m, [1, 2, 3], ctr)
        self.assertGreater(ctr["flops"], 0)
        base = ctr["flops"]
        forward_all(self.m, [1, 2, 3, 4], ctr)
        self.assertGreater(ctr["flops"], base)


class EquivalenceTests(unittest.TestCase):
    def test_logits_match_exactly(self):
        m = Model(vocab=12, dim=8, heads=2, seed=3)
        tokens = [1, 4, 2, 5, 0]
        full = forward_all(m, tokens)
        c = Cache(m)
        cached = []
        for t in tokens:
            cached = cache_token(m, c, t)
        self.assertEqual(full, cached)  # same op order: bitwise identical

    def test_greedy_generations_identical(self):
        m = Model(vocab=12, dim=8, heads=2, seed=3)
        tf, _ = generate_full(m, [1, 4, 2], 8)
        tc, _ = generate_cached(m, [1, 4, 2], 8)
        self.assertEqual(tf, tc)

    def test_identical_across_configs(self):
        for seed, dim, heads in [(0, 4, 1), (5, 12, 3), (7, 8, 4)]:
            m = Model(vocab=9, dim=dim, heads=heads, seed=seed)
            tf, _ = generate_full(m, [2, 0], 5)
            tc, _ = generate_cached(m, [2, 0], 5)
            self.assertEqual(tf, tc, f"config seed={seed}")

    def test_cached_is_cheaper(self):
        m = Model(vocab=12, dim=8, heads=2, seed=3)
        _, ff = generate_full(m, [1, 4, 2], 8)
        _, fc = generate_cached(m, [1, 4, 2], 8)
        self.assertLess(fc, ff)


if __name__ == "__main__":
    unittest.main()
