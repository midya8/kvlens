"""A tiny causal attention model with two decoding paths: full recompute and
KV-cached incremental decoding.

Weights are seeded, not trained: the point is the decoding machinery. The
full path recomputes attention over the whole prefix for every new token;
the cached path stores each position's K and V once and attends over the
cache. Both must produce the same next-token distribution, and the module
counts matmul FLOPs in each path so the savings are measured, not assumed.
"""

from __future__ import annotations

import math
import random

Matrix = list[list[float]]


def matmul(a: Matrix, b: Matrix, counter: dict | None = None) -> Matrix:
    bt = list(zip(*b))
    if counter is not None:
        counter["flops"] += 2 * len(a) * len(b) * len(b[0])
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def softmax(row: list[float]) -> list[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def positional_encoding(length: int, dim: int) -> Matrix:
    return [
        [math.sin(p / 10000.0 ** (2 * (i // 2) / dim)) if i % 2 == 0
         else math.cos(p / 10000.0 ** (2 * (i // 2) / dim))
         for i in range(dim)]
        for p in range(length)
    ]


class Model:
    """One causal multi-head attention block plus an output head."""

    def __init__(self, vocab: int, dim: int, heads: int, seed: int = 0) -> None:
        if heads < 1 or dim % heads != 0:
            raise ValueError("dim must be divisible by heads, heads >= 1")
        if vocab < 2:
            raise ValueError("vocab must be >= 2")
        rng = random.Random(seed)
        self.vocab, self.dim, self.heads = vocab, dim, heads
        self.head_dim = dim // heads
        scale = 1.0 / math.sqrt(dim)
        init = lambda r, c: [[(rng.random() - 0.5) * 2 * scale for _ in range(c)] for _ in range(r)]
        self.embed = init(vocab, dim)
        self.wq = [init(dim, self.head_dim) for _ in range(heads)]
        self.wk = [init(dim, self.head_dim) for _ in range(heads)]
        self.wv = [init(dim, self.head_dim) for _ in range(heads)]
        self.wo = init(dim, dim)
        self.wout = init(dim, vocab)
        self.bout = [0.0] * vocab


class Cache:
    """Per-head cached K and V matrices, one row per processed position."""

    def __init__(self, model: Model) -> None:
        self.k: list[Matrix] = [[] for _ in range(model.heads)]
        self.v: list[Matrix] = [[] for _ in range(model.heads)]
        self.length = 0


def _embed_row(model: Model, token: int, pos: int) -> list[float]:
    pe = positional_encoding(pos + 1, model.dim)[pos]
    return [model.embed[token][i] + pe[i] for i in range(model.dim)]


def _project_row(model: Model, x: list[float], w: Matrix) -> list[float]:
    return [sum(x[d] * w[d][c] for d in range(model.dim)) for c in range(len(w[0]))]


def _attend_query(model: Model, q_row: list[float], k: Matrix, v: Matrix) -> list[float]:
    """One query row against cached/full K,V. Same op order in both paths."""
    scale = 1.0 / math.sqrt(model.head_dim)
    scores = [sum(q_row[d] * kj[d] for d in range(model.head_dim)) * scale for kj in k]
    probs = softmax(scores)
    return [sum(probs[j] * v[j][c] for j in range(len(v))) for c in range(model.head_dim)]


def forward_all(model: Model, tokens: list[int], counter: dict | None = None) -> list[float]:
    """Full-recompute forward: logits at the final position, computing every
    position's attention. This is the baseline an incremental decoder avoids."""
    length = len(tokens)
    x = [_embed_row(model, t, pos) for pos, t in enumerate(tokens)]
    head_out: Matrix = []
    for h in range(model.heads):
        q = matmul(x, model.wq[h], counter)
        k = matmul(x, model.wk[h], counter)
        v = matmul(x, model.wv[h], counter)
        outs = []
        for i in range(length):
            if counter is not None:
                counter["flops"] += 2 * (i + 1) * model.head_dim * 2
            outs.append(_attend_query(model, q[i], k[: i + 1], v[: i + 1]))
        head_out.append(outs)
    last = [head_out[h][length - 1][c] for h in range(model.heads) for c in range(model.head_dim)]
    if counter is not None:
        counter["flops"] += 2 * model.dim * model.dim + 2 * model.dim * model.vocab
    out = [sum(last[d] * model.wo[d][c] for d in range(model.dim)) for c in range(model.dim)]
    return [sum(out[d] * model.wout[d][c] for d in range(model.dim)) + model.bout[c]
            for c in range(model.vocab)]


def cache_token(model: Model, cache: Cache, token: int, counter: dict | None = None) -> list[float]:
    """Incremental step: append one token, return logits at its position,
    computing K/V for this token only and attending over the cache."""
    x = _embed_row(model, token, cache.length)
    head_out: list[float] = []
    for h in range(model.heads):
        cache.k[h].append(_project_row(model, x, model.wk[h]))
        cache.v[h].append(_project_row(model, x, model.wv[h]))
        if counter is not None:
            counter["flops"] += 2 * model.dim * model.head_dim * 3  # q, k, v
            counter["flops"] += 2 * (cache.length + 1) * model.head_dim * 2
        q_row = _project_row(model, x, model.wq[h])
        head_out.extend(_attend_query(model, q_row, cache.k[h], cache.v[h]))
    cache.length += 1
    if counter is not None:
        counter["flops"] += 2 * model.dim * model.dim + 2 * model.dim * model.vocab
    out = [sum(head_out[d] * model.wo[d][c] for d in range(model.dim)) for c in range(model.dim)]
    return [sum(out[d] * model.wout[d][c] for d in range(model.dim)) + model.bout[c]
            for c in range(model.vocab)]



def prefill(model: Model, cache: Cache, prompt: list[int], counter: dict | None = None) -> list[float]:
    """Batched prompt processing: compute every prompt position's K/V in one
    pass per head, store them in the cache, and return logits at the final
    prompt position. This is the prefill phase of a real KV-cached decoder;
    the incremental cache_token steps then handle generated tokens only."""
    if not prompt:
        raise ValueError("prefill needs a non-empty prompt")
    length = len(prompt)
    x = [_embed_row(model, t, pos) for pos, t in enumerate(prompt)]
    head_out: Matrix = []
    for h in range(model.heads):
        q = matmul(x, model.wq[h], counter)
        k = matmul(x, model.wk[h], counter)
        v = matmul(x, model.wv[h], counter)
        for i in range(length):
            cache.k[h].append(k[i])
            cache.v[h].append(v[i])
            if counter is not None:
                counter["flops"] += 2 * (i + 1) * model.head_dim * 2
            head_out_row = _attend_query(model, q[i], k[: i + 1], v[: i + 1])
        head_out.append(head_out_row)
    cache.length += length
    last = [head_out[h][c] for h in range(model.heads) for c in range(model.head_dim)]
    if counter is not None:
        counter["flops"] += 2 * model.dim * model.dim + 2 * model.dim * model.vocab
    out = [sum(last[d] * model.wo[d][c] for d in range(model.dim)) for c in range(model.dim)]
    return [sum(out[d] * model.wout[d][c] for d in range(model.dim)) + model.bout[c]
            for c in range(model.vocab)]

def generate_full(model: Model, prompt: list[int], steps: int) -> tuple[list[int], int]:
    """Greedy generation recomputing the full prefix each step."""
    counter = {"flops": 0}
    tokens = list(prompt)
    for _ in range(steps):
        logits = forward_all(model, tokens, counter)
        tokens.append(max(range(model.vocab), key=lambda c: logits[c]))
    return tokens, counter["flops"]


def generate_cached(model: Model, prompt: list[int], steps: int) -> tuple[list[int], int]:
    """Greedy generation with a KV cache: batched prefill over the prompt,
    then one incremental step per new token."""
    counter = {"flops": 0}
    cache = Cache(model)
    tokens = list(prompt)
    logits = prefill(model, cache, tokens, counter)
    for i in range(steps):
        nxt = max(range(model.vocab), key=lambda c: logits[c])
        tokens.append(nxt)
        if i + 1 < steps:
            logits = cache_token(model, cache, nxt, counter)
    return tokens, counter["flops"]
