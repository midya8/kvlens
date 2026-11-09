"""Command-line interface for KVLens."""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import equivalence_sweep, savings_curve
from .model import Model, generate_cached, generate_full


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vocab", type=int, default=16)
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--prompt", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kvlens", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="greedy generation down both paths")
    _add_eval_args(p_gen)
    p_gen.add_argument("--require-identical", action="store_true",
                       help="exit 1 unless both paths emit the same tokens")
    p_gen.add_argument("--min-savings", type=float, default=None,
                       help="exit 1 unless the cached path saves at least this fraction of FLOPs")

    p_eq = sub.add_parser("equivalence", help="logit-level equivalence sweep")
    p_eq.add_argument("--tol", type=float, default=1e-9)

    p_curve = sub.add_parser("curve", help="FLOP savings over generation length")
    _add_eval_args(p_curve)
    p_curve.add_argument("--min-savings", type=float, default=None)

    p_json = sub.add_parser("json", help="full report as JSON")
    _add_eval_args(p_json)

    args = parser.parse_args(argv)

    if hasattr(args, "steps") and args.steps < 1:
        raise ValueError("original CLI requires steps >= 1; kvlens.audit supports a zero-work budget")

    if args.cmd == "generate":
        model = Model(vocab=args.vocab, dim=args.dim, heads=args.heads, seed=args.seed)
        prompt = [i % args.vocab for i in range(args.prompt)]
        toks_f, flops_f = generate_full(model, prompt, args.steps)
        toks_c, flops_c = generate_cached(model, prompt, args.steps)
        identical = toks_f == toks_c
        saved = 1.0 - flops_c / flops_f
        print(f"prompt {prompt} + {args.steps} greedy steps")
        print(f"full path:   {flops_f} matmul FLOPs")
        print(f"cached path: {flops_c} matmul FLOPs ({saved:.1%} saved)")
        print(f"tokens identical: {identical}")
        if args.require_identical and not identical:
            print("GATE FAILED: paths diverge")
            return 1
        if args.min_savings is not None and saved < args.min_savings:
            print(f"GATE FAILED: savings {saved:.1%} < {args.min_savings:.1%}")
            return 1
        return 0

    if args.cmd == "equivalence":
        rows = equivalence_sweep(tol=args.tol)
        worst = max(rows, key=lambda r: r["max_abs_diff"])
        for r in rows:
            status = "ok" if r["match"] else "MISMATCH"
            print(f"[{r['config']}] length {r['length']}: max |diff| {r['max_abs_diff']:.2e} {status}")
        print(f"worst diff across sweep: {worst['max_abs_diff']:.2e} (tol {args.tol:.0e})")
        return 0 if all(r["match"] for r in rows) else 1

    if args.cmd == "curve":
        rows = savings_curve(vocab=args.vocab, dim=args.dim, heads=args.heads,
                             seed=args.seed, prompt=args.prompt,
                             lengths=(2, 4, 8, 16, args.steps))
        for r in rows:
            print(f"{r['new_tokens']:>4} new tokens: full {r['full_flops']:>8} | cached {r['cached_flops']:>8} | saved {r['saved']:.1%}")
        worst = min(r["saved"] for r in rows)
        if args.min_savings is not None and worst < args.min_savings:
            print(f"GATE FAILED: minimum savings {worst:.1%} < {args.min_savings:.1%}")
            return 1
        return 0

    if args.cmd == "json":
        model = Model(vocab=args.vocab, dim=args.dim, heads=args.heads, seed=args.seed)
        prompt = [i % args.vocab for i in range(args.prompt)]
        toks_f, flops_f = generate_full(model, prompt, args.steps)
        toks_c, flops_c = generate_cached(model, prompt, args.steps)
        report = {
            "vocab": args.vocab, "dim": args.dim, "heads": args.heads, "seed": args.seed,
            "steps": args.steps,
            "tokens_identical": toks_f == toks_c,
            "full_flops": flops_f,
            "cached_flops": flops_c,
            "savings": round(1.0 - flops_c / flops_f, 4),
            "equivalence": equivalence_sweep(),
            "curve": savings_curve(vocab=args.vocab, dim=args.dim, heads=args.heads,
                                   seed=args.seed, prompt=args.prompt,
                                   lengths=(2, 4, 8, 16, args.steps)),
        }
        print(json.dumps(report, indent=2))
        return 0

    return 2


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (ValueError, OverflowError) as exc:
        print(f"kvlens: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
