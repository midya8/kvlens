# KVLens

KV-cached incremental decoding built from scratch in pure Python, verified token-for-token against full recompute, with the FLOP savings measured rather than assumed.

KVLens implements a tiny causal multi-head attention block (seeded weights - the point is the decoding machinery, not training) and runs generation down two paths:

- **Full recompute** - the naive path: for every new token, recompute Q/K/V and attention over the entire prefix.
- **KV cache** - the production path: a batched prefill computes each prompt position's K and V once; every decode step computes K/V for the new token only and attends over the cache.

The two paths share the same operations in the same order, so their logits are **bitwise identical** - the equivalence sweep reports a max absolute logit difference of exactly 0.0 across model configurations and prefix lengths. Greedy generations match token for token. A matmul FLOP counter in both paths measures the savings directly.

## Measured results

Default fixture (vocab 16, dim 8, 2 heads, prompt 4, seeded weights), from `kvlens curve`:

| new tokens | full FLOPs | cached FLOPs | saved |
|---|---|---|---|
| 2 | 5,024 | 3,168 | 36.9% |
| 4 | 12,352 | 5,120 | 58.6% |
| 8 | 34,944 | 9,408 | 73.1% |
| 16 | 119,040 | 19,520 | 83.6% |
| 32 | 500,224 | 45,888 | 90.8% |

Savings grow with generation length and shrink with a longer prompt, exactly as the attention cost model predicts. At 1 generated token the paths cost the same: prefill is the full forward, and caching only pays off once decoding continues.

## What is verified

- **Bitwise logit equivalence** between the paths across two model configurations and five prefix lengths (`kvlens equivalence`); greedy generations identical across further configurations in the test suite.
- **Measured savings**: CI gates require at least 85% FLOP savings at 32 generated tokens on the default fixture and a monotonically increasing savings curve.
- The counter counts matmul FLOPs only (Q/K/V projections, attention scores/weighted sums, output projections); it excludes embeddings, softmax, and adds. The saved fraction is stated for that counted cost, not for wall-clock time - pure-Python inner loops are dominated by interpreter overhead, and this project makes no latency claim.

## Usage

```bash
python -m kvlens.cli generate --steps 32 --require-identical --min-savings 0.85
python -m kvlens.cli equivalence            # bitwise logit-equivalence sweep
python -m kvlens.cli curve                  # FLOP savings over generation length
python -m kvlens.cli json                   # full machine-readable report
```

`generate --require-identical` exits non-zero if the paths diverge; `--min-savings` gates the measured savings. Both run on every CI push.

## Layout

- `kvlens/model.py` - seeded causal attention model, `Cache`, batched `prefill`, incremental `cache_token`, and both generation paths with FLOP counters.
- `kvlens/analysis.py` - logit-equivalence checks, the configuration/length sweep, and the savings curve.
- `kvlens/cli.py` - commands above.
- `tests/` - 28 unit tests covering shapes, seeded determinism, bitwise and greedy equivalence across configs, cost ordering, and every CLI gate.

## Limits

- One attention block, no training: weights are seeded RNG draws, so generated token sequences are not meaningful language - the project verifies decoding machinery, not model quality.
- Greedy decoding only; no batching across sequences, no paged/evicted caches.
- FLOP counts cover matmuls as described above; memory-bandwidth effects (the real reason KV caching is fast) are discussed, not measured.
