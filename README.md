# KVLens

KV-cached incremental decoding from scratch, with a complete offline attention
inspector and a stronger final-query-only baseline. Same logits on checked
inputs; explicit counted operations, not a latency benchmark.

[Live attention and cache inspector](https://midya8.github.io/kvlens/) ·
[State contract, equations and attribution](CACHE.md)

## A stronger comparison, every tensor

The historical baseline computes every past query's attention even though this
single block only uses the final row. The new no-cache path computes only that
query. An optimized prefill uses the same shortcut; the old pair stays intact.
For seed 1, prompt[0,1,2,3] and 8 predictions, old-full/final-query/historical-cache/
optimized-cache totals are 34,944/21,376/9,408/8,832 counted operations. The old pair
saves 73.08%; optimized cache versus the stronger no-cache path saves 58.68%.
These are named comparisons, not an isolated causal effect or measured speedup.

The independent browser computes all seven seeded model weight sets, every
prediction, head, key/value row, score, softmax weight, value contribution,
output/logit term and before/after cache. Full JSON and model exports, editable
prompts/budgets, per-step operation ledgers and retained-state counts make the
whole path inspectable. Weight arrays are inputs, not saved inference results.

Fixed a real state bug: prefill on an occupied cache used to append tokens at
restarted positions. It now rejects before changing state. Cache ownership,
shape/token guards and deep independent forks cover lifecycle mistakes. Weights
must stay fixed; same-object weight mutation is not fingerprint-detected and
these guards are not a security boundary. Zero-token generation does no work.

55 Python tests retain all 28 originals; three Node/actual-Chrome tests compare 32
complete reports,97 predictions and 66,940 numeric entries. Chrome traverses 227
head-step views/1,279 attention-position rows, all model-weight tables, two JSON
exports, keyboard and 320/390/1280 layouts. An isolated installed wheel verifies
five module origins, all reports, original curve/equivalence, lifecycle checks,
the audit CLI and every original CLI mode. Finite tests are not a universal proof.

```bash
python -m kvlens.audit --input examples/audit.json
python scripts/verify_regeneration.py
node --test tests/*.cjs
bash examples/verify_wheel.sh
```

Audit input is bounded, strict JSON; errors exit 2 without partial output. All
original positive-budget results below remain unchanged. Ideal K/V binary 64
payload counts are not actual Python/browser memory or measured bandwidth.

## Original two-path fixture

KVLens implements a tiny causal multi-head attention block (seeded weights - the point is the decoding machinery, not training) and runs generation down two paths:

- **Full recompute** - the naive path: for every new token, recompute Q/K/V and attention over the entire prefix.
- **KV cache** - the original cached path: a batched prefill computes each prompt position's K and V once; every decode step computes K/V for the new token only and attends over the cache.

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
- The counter counts matmul FLOPs only (Q/K/V projections, attention scores/weighted sums, output projections); it excludes embeddings, softmax and non-matmul adds (dot-product accumulation is counted). The saved fraction is stated for that counted cost, not for wall-clock time - pure-Python inner loops are dominated by interpreter overhead, and this project makes no latency claim.

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
