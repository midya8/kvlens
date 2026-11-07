import io
import json
import unittest
from contextlib import redirect_stdout

from kvlens.cli import main


def run_cli(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(args)
    return code, buf.getvalue()


class GenerateTests(unittest.TestCase):
    def test_reports_both_paths(self):
        code, out = run_cli(["generate", "--steps", "8"])
        self.assertEqual(code, 0)
        self.assertIn("full path:", out)
        self.assertIn("cached path:", out)
        self.assertIn("tokens identical: True", out)

    def test_require_identical_passes(self):
        code, _ = run_cli(["generate", "--steps", "8", "--require-identical"])
        self.assertEqual(code, 0)

    def test_min_savings_passes_at_reachable_bar(self):
        code, out = run_cli(["generate", "--steps", "32", "--min-savings", "0.5"])
        self.assertEqual(code, 0, out)

    def test_min_savings_fails_at_impossible_bar(self):
        code, out = run_cli(["generate", "--steps", "4", "--min-savings", "0.99"])
        self.assertEqual(code, 1)
        self.assertIn("GATE FAILED", out)


class EquivalenceTests(unittest.TestCase):
    def test_sweep_passes(self):
        code, out = run_cli(["equivalence"])
        self.assertEqual(code, 0, out)
        self.assertIn("worst diff", out)

    def test_tight_tolerance_still_passes(self):
        code, _ = run_cli(["equivalence", "--tol", "1e-15"])
        self.assertEqual(code, 0)  # same op order: diff is exactly 0


class CurveTests(unittest.TestCase):
    def test_prints_rows(self):
        code, out = run_cli(["curve"])
        self.assertEqual(code, 0)
        self.assertIn("saved", out)

    def test_gate(self):
        self.assertEqual(run_cli(["curve", "--min-savings", "0.3"])[0], 0)
        self.assertEqual(run_cli(["curve", "--min-savings", "0.95"])[0], 1)


class JsonTests(unittest.TestCase):
    def test_report_shape(self):
        code, out = run_cli(["json", "--steps", "8"])
        self.assertEqual(code, 0)
        r = json.loads(out)
        for key in ("tokens_identical", "full_flops", "cached_flops", "savings", "equivalence", "curve"):
            self.assertIn(key, r)
        self.assertTrue(r["tokens_identical"])


if __name__ == "__main__":
    unittest.main()
