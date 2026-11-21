"""Counter-based randomness (docs §7, decision L5).

Every random draw is keyed by (world seed, mechanism name, tick). Re-running a tick, a
partial rerun from a checkpoint, or a counterfactual twin gets exactly the same stream,
so exogenous noise is shared between factual and twin runs.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _key(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=16).digest(), "little")


def stream(seed: int, name: str, counter: int = 0) -> np.random.Generator:
    """Independent generator for (seed, name, counter)."""
    return np.random.Generator(np.random.Philox(key=_key(seed, name), counter=[counter, 0, 0, 0]))
