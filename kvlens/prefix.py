"""Independent prefix branches of already processed K/V state.

Only fixed-weight, append-only position encodings are supported. This copies
stored rows; it neither recomputes projections nor edits the original cache.
"""
from .model import Cache, _validate_cache


def prefix_cache(model, cache, length):
    """Copy a processed prefix into a new independent cache.

    ``length`` counts processed tokens, not emitted predictions. A generation
    routine may return one final emitted token that is not in its cache.
    The caller remains responsible for associating this state with its tokens
    and keeping the model weights fixed, as for the existing cache API.
    """
    _validate_cache(model, cache)
    if type(length) is not int or not 0 <= length <= cache.length:
        raise ValueError('prefix length must be an integer in [0, cache.length]')
    child = Cache(model)
    child.k = [[list(row) for row in head[:length]] for head in cache.k]
    child.v = [[list(row) for row in head[:length]] for head in cache.v]
    child.length = length
    return child
