"""Equivalence checks and FLOP-savings measurement for the two decode paths."""

from __future__ import annotations

import random

from .model import Cache, Model, cache_token, forward_all, generate_cached, generate_full


def logits_match(a: list[float], b: list[float], tol: float = 1e-9) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def check_equivalence(model: Model, tokens: list[int], tol: float = 1e-9) -> dict:
    """Run both paths over the same prefix and compare the final-position
    logits. Identical op order makes them match to floating-point noise."""
    full = forward_all(model, tokens)
    cache = Cache(model)
    cached: list[float] = []
    for t in tokens:
        cached = cache_token(model, cache, t)
    diff = max(abs(x - y) for x, y in zip(full, cached))
    return {"length": len(tokens), "max_abs_diff": diff, "match": diff <= tol}


def equivalence_sweep(vocab: int = 12, dim: int = 8, heads: int = 2,
                      lengths: tuple[int, ...] = (2, 3, 5, 8, 13),
                      configs: tuple[tuple[int, int, int], ...] = ((3, 8, 2), (4, 12, 3)),
                      tol: float = 1e-9) -> list[dict]:
    rng = random.Random(0)
    results = []
    for cfg_seed, cfg_dim, cfg_heads in configs:
        model = Model(vocab=vocab, dim=cfg_dim, heads=cfg_heads, seed=cfg_seed)
        for n in lengths:
            tokens = [rng.randrange(vocab) for _ in range(n)]
            r = check_equivalence(model, tokens, tol)
            r["config"] = f"seed={cfg_seed},dim={cfg_dim},heads={cfg_heads}"
            results.append(r)
    return results


def savings_curve(vocab: int = 16, dim: int = 8, heads: int = 2, seed: int = 1,
                  prompt: int = 4, lengths: tuple[int, ...] = (2, 4, 8, 16, 32)) -> list[dict]:
    """Measured matmul FLOPs of both paths over greedy generations of
    increasing length, plus the saved fraction."""
    model = Model(vocab=vocab, dim=dim, heads=heads, seed=seed)
    rows = []
    for steps in lengths:
        toks_f, flops_f = generate_full(model, list(range(prompt)), steps)
        toks_c, flops_c = generate_cached(model, list(range(prompt)), steps)
        assert toks_f == toks_c
        rows.append({
            "new_tokens": steps,
            "full_flops": flops_f,
            "cached_flops": flops_c,
            "saved": round(1.0 - flops_c / flops_f, 4),
        })
    return rows
