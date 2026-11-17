# kvlens

A KV cache for a single attention block, written out in plain Python so you can count what it saves. The weights are random (seeded), since the point is the decoding mechanics, not the text it produces.

Generation runs two ways: the naive one that recomputes attention over the whole prefix for every new token, and the cached one that keeps each position's keys and values around and only computes the new token's. Both do the same operations in the same order, so the logits match bit for bit, and a counter tallies matmul FLOPs for each.

```bash
python -m kvlens.cli generate --steps 32 --require-identical --min-savings 0.85
python -m kvlens.cli curve
python -m kvlens.cli equivalence
```

With a 4-token prompt (vocab 16, dim 8, 2 heads):

| new tokens | full FLOPs | cached FLOPs | saved |
|---|---|---|---|
| 2 | 5,024 | 3,168 | 36.9% |
| 8 | 34,944 | 9,408 | 73.1% |
| 32 | 500,224 | 45,888 | 90.8% |

Savings grow the longer you generate. At one new token there's no difference, because the prefill is already a full forward pass. These are counted FLOPs, not wall-clock time; in pure Python the interpreter overhead swamps everything, and the real-world win from caching is mostly memory bandwidth, which this doesn't model.

A fairer comparison came later: the naive path computes attention for every past query, but only the last row matters. Making the uncached path compute just that last query brings the cache's saving at 8 tokens down from 73% to about 59%. CACHE.md goes through both.

## Branching a cache

You can copy the first n positions of a cache and continue from there, like rewinding to a shared prefix:

```python
from kvlens.model import Model, Cache, prefill, cache_token
from kvlens.prefix import prefix_cache

model = Model(vocab=8, dim=8, heads=2, seed=1)
root = Cache(model)
prefill(model, root, [1, 2, 3, 4])
branch = prefix_cache(model, root, 2)
logits = cache_token(model, branch, 7)   # context is now [1, 2, 7]
```

Writing this turned up a real bug: calling prefill on a cache that already had tokens used to append them at the wrong positions. It now refuses.

There's a browser page that shows every attention score and cache row step by step: https://midya8.github.io/kvlens/

Tests: `python -m unittest discover -s tests` (59 tests). Greedy decoding only, one block, no batching.
