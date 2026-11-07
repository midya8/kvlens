import unittest

from kvlens.analysis import check_equivalence, equivalence_sweep, logits_match, savings_curve
from kvlens.model import Model


class LogitsMatchTests(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(logits_match([1.0, 2.0], [1.0, 2.0]))

    def test_within_tol(self):
        self.assertTrue(logits_match([1.0], [1.0 + 1e-10], tol=1e-9))
        self.assertFalse(logits_match([1.0], [1.0 + 1e-6], tol=1e-9))

    def test_length_mismatch(self):
        self.assertFalse(logits_match([1.0], [1.0, 2.0]))


class EquivalenceTests(unittest.TestCase):
    def test_single_check(self):
        m = Model(vocab=12, dim=8, heads=2, seed=3)
        r = check_equivalence(m, [1, 4, 2, 5])
        self.assertTrue(r["match"])
        self.assertEqual(r["max_abs_diff"], 0.0)

    def test_sweep_all_match(self):
        rows = equivalence_sweep()
        self.assertEqual(len(rows), 10)  # 2 configs x 5 lengths
        self.assertTrue(all(r["match"] for r in rows))


class SavingsCurveTests(unittest.TestCase):
    def test_rows_and_monotonic_savings(self):
        rows = savings_curve()
        self.assertEqual(len(rows), 5)
        saved = [r["saved"] for r in rows]
        self.assertTrue(all(a < b for a, b in zip(saved, saved[1:])), "savings grow with length")
        self.assertGreater(saved[-1], 0.5)

    def test_cached_never_more_expensive(self):
        rows = savings_curve(vocab=8, dim=4, heads=1, lengths=(1, 2, 4))
        self.assertTrue(all(r["cached_flops"] <= r["full_flops"] for r in rows))


if __name__ == "__main__":
    unittest.main()
