"""KVLens: KV-cached incremental decoding from scratch, verified against full recompute."""

from .model import Cache, Model, cache_token, forward_all, generate_cached, generate_full, prefill
from .analysis import check_equivalence, equivalence_sweep, savings_curve

__version__ = "0.1.0"

__all__ = [
    "Cache", "Model", "cache_token", "forward_all", "generate_cached", "generate_full",
    "prefill", "check_equivalence", "equivalence_sweep", "savings_curve",
]
