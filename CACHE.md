# KVLens attention and cache method

KVLens is a single untrained causal attention block. It is an implementation of
standard attention and KV reuse, not a new inference algorithm or useful language
model. The original 28 tests, `analysis.py`, demo and positive-budget fixture are
retained. Original file hashes and commit are in `kvlens/data/original-hashes.json`.
`model.py` adds validation, ownership and branch guards, zero-work generation and
two stronger paths. `cli.py` converts invalid parameters to exit 2 with stderr,
and rejects a zero budget in its historical ratio commands rather than dividing
by zero. The separate audit accepts zero as no work.

## Four paths, one model

For n input positions, width d, vocabulary v and head width d/h:

1. Historical `forward_all`: project every Q/K/V, compute every causal attention
   row, but apply the output and vocabulary projections only at the last row.
2. `forward_last`: compute K/V for every input, only the last Q and attention row.
   This is a stronger no-cache baseline, not a deliberately weakened control.
3. Historical cache: original all-row prefill, then one `cache_token` per newly
   processed input. Positive-budget operation order and counted costs stay intact.
4. Optimized cache: final-query-only prefill followed by the same incremental step.

All four evaluate the same logits at every prediction under native operation
order on checked inputs. Earliest vocabulary index wins exact greedy ties.
The audit compares logits, not just chosen tokens. Each browser path is computed
separately from all source weight arrays, not replaced with saved logits or a
selected summary. The Python RNG's seven exported weight sets are inputs to JS;
JS does not pretend to reproduce Python's RNG. Python accepts other bounded
seeds. Native/browser comparisons use normalized tolerance, not a cross-runtime
bitwise claim. Near-zero greedy margins can make tolerance agreement insufficient
to imply matching decisions on arbitrary unseen cases; all tested actual decisions
are checked separately.

The stronger path is valid because the K/V of this one block project token and
position inputs, not earlier attention outputs. A multilayer transformer still
needs earlier layer representations for later-layer keys/values. Do not generalize
the shortcut into skipping that work.

## Counted operation ledger

A length-k dot product is charged 2 k, including multiplication and accumulation.
Per input length n, across all heads:

| component | old full | final-query no cache | incremental append |
|---|---:|---:|---:|
| Q | 2 nd² | 2 d² | 2 d² |
| K | 2 nd² | 2 nd² | 2 d² |
| V | 2 nd² | 2 nd² | 2 d² |
| attention scores | dn(n+1) | 2 nd | 2 nd |
| weighted values | dn(n+1) | 2 nd | 2 nd |
| output projection | 2 d² | 2 d² | 2 d² |
| vocabulary projection | 2 dv | 2 dv | 2 dv |

These reconcile with the actual counters at every step. They exclude embedding
lookup/addition, sinusoidal functions, score scaling, softmax, bias adds,
validation, copying, JSON construction and report/UI work. They count no time,
cache traffic, memory bandwidth, hardware instructions, GPU kernels or throughput.
The original README's phrase "excludes adds" was too broad: dot-product
accumulation is counted; non-matmul adds are excluded.

For the default eight-prediction control (v=16,d=8,h=2,prompt [0,1,2,3],seed 1):

| path | counted total |
|---|---:|
| historical full | 34,944 |
| final-query no-cache | 21,376 |
| historical cache | 9,408 |
| optimized cache | 8,832 |

The historical pair saves 73.08%; optimized cache versus the stronger no-cache
path saves 58.68%. Different baselines and prefill policies are named, not
conflated into an isolated causal effect. All four component tables are available.
The original 32-token historical comparison remains 500,224 versus 45,888 (90.8%
to one decimal); the original CI gate is unchanged. At one prediction there is no
reuse within either matched pair. At zero predictions all totals are zero and
savings ratios are undefined, never a fabricated percentage.

## State and positions

The cache stores per-head K/V rows and a processed-position count. Prefill accepts
only an empty cache. Appending uses the current length as the absolute sinusoidal
position. Previously, calling prefill on occupied state appended new rows but
restarted positions at zero, silently mixing incompatible representations. It is
now rejected before mutations or counter updates. A cache belongs to a specific
model object; a different object, even with identical configuration, is rejected.
Malformed shapes, nonfinite cache entries, empty/invalid token sequences and
negative/boolean/noninteger generation lengths are rejected. Independent forks
deep-copy every row; append or row edits in a branch do not alter siblings.

Weights must stay fixed for a cache's lifetime. Same-object weight mutation is
not fingerprint-detected. Public arrays remain mutable: well-shaped malicious
or incorrect values cannot be authenticated by shape checks. Guards are not a
security or isolation system and cannot validate the semantic origin of a
manually edited cache. Clear/rebuild after changing weights or prefix identity.

To generate G>0 tokens from a prompt of length L, the decoder processes L+G-1
positions. The final emitted token is not fed back because no further prediction
is requested. For G=0 it performs no forward pass, allocates no cache and returns
a copy of the prompt. This convention is explicit; external descriptions may
choose to process/store the last token for future continuation, which is a
different stopping contract, not an inconsistency in the arithmetic here.

Retained K/V state has 2 nd elements. At binary 64's eight bytes each, ideal packed
payload is 16 nd bytes. This is not Python-list/object memory, JS heap, tensor
allocator overhead, temporary memory, measured bandwidth or actual peak process
memory. Reports include full cache before/after states, not just a length label.

## Strict input, full exports and checks

`python -m kvlens.audit --input examples/audit.json` or `--input -` accepts exactly
`vocab,dim,heads,seed,prompt,steps`. Bounds: vocab2..32,dim2..16,heads1..8 dividing
dim,seed 0..1000000,prompt 1..20 in-vocabulary integers,steps 0..12. Booleans do not
count as integers. Duplicate keys, nonfinite/exponent-overflow values, unknown
fields, depth over 8 and input over 100000 UTF-8 bytes fail with exit 2, stderr and no
partial output. Browser edits clear reports and disable both downloads until a
valid recomputation. The browser supports only the seven supplied model IDs,
with all supported models named in its weight export; it reports unsupported IDs
instead of using the wrong weights.

Every output includes all predictions, input embeddings, head Q/K/V, score
terms, scores, probabilities, weighted value contributions, joined heads,
output-projection terms, every vocabulary-logit term, all four counted ledgers,
cumulative costs, margins and complete before/after caches. Full audit and full
model are separate downloadable JSON files. Reference data stores all weight
inputs and 32 complete input specifications plus pinned tokens/costs; it does not
substitute static intermediate tensors for computation. Native full reports are
recomputed for every test comparison.

55 Python tests include all 28 unchanged originals,36 additional randomized model
configurations, hand-computable uniform attention, operation-count reconciliation,
cache guards, forks, strict schemas and CLI errors. Three Node/actual-Chrome tests
compare 32 complete native reports/97 predictions and 66,940 numeric entries.
Actual Chrome traverses 227 head-step views and 1,279 attention-position rows,
checks every model weight table, full JSON exports, keyboard, invalid clearing
and 320/390/1280 layouts. These are finite tests, not a universal proof.

Native/JS normalized tolerance is1e-11; complete fixture/model regeneration uses
1e-12 and restores original checked-in bytes. Python3.10-3.13 may differ in last
bits due to floating-point summation, so persisted float fixtures are not compared
by textual equality. An isolated installed wheel checks five module origins,
all 32 reports, original complete curve and equivalence, forks/guards, audit CLI
and all four original CLI modes outside checkout.

## Sources and attribution

- Vaswani et al., *Attention Is All You Need*,2017,sections 3.2 and 3.5:
  https://arxiv.org/abs/1706.03762v7
  Scaled dot-product/multi-head attention and sinusoidal positions. Project does
  not reproduce its trained transformer, architecture or translation results.
- Hugging Face, *How caching works*:
  https://huggingface.co/docs/transformers/cache_explanation
  Only the new query is needed in cached inference; KV grows per processed
  position/layer. This project does not implement Transformers cache APIs.
- NVIDIA,*Mastering LLM Techniques: Inference Optimization*,2023:
  https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
  Prefill/decode distinction, memory-bound context and cache footprint. No GPU
  performance or vendor benchmark is measured or reproduced here.
- Pieter Delobelle,*NanoGPT-inference - KV Caching*,2025:
  https://pieter.ai/blog/2025/nanogpt-inference---kv-caching/
  New query versus reused K/V and prefill/decode exposition. Its throughput
  measurements are not KVLens results.
- Bruno Magalhaes,*Prefill vs decode, roofline, KV cache, batching and disaggregation*,2025:
  https://brunomaga.github.io/inference-prefill-decode-roofline-cache
  Arithmetic versus memory traffic, state positions and cache/prefix distinctions.
  Roofline costs and hardware numbers are not adopted as toy-Python measurements.
- Brown University,CS 146 transformer teaching notes:
  https://cs.brown.edu/courses/cs146/transformer.html
  Standard causal/attention implementation context. No course work or training
  claim is implied. The project implementations and original model remain local.

All methods are standard. Model weights are the original project's seeded random
inputs, not trained public weights or private data. No paid inference service,
private records, external model output or invented attribution is used.
