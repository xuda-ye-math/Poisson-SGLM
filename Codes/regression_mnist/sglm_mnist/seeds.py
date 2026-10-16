"""Stable, explicit random-stream derivation for experiment phases."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

import numpy as np


SEED_SCHEMA_VERSION = 1
BIT_GENERATOR = "PCG64DXSM"


def derive_seed(namespace: str, *components: object) -> Tuple[int, Dict[str, Any]]:
    """Derive a 128-bit seed without Python's process-dependent ``hash``."""

    if not namespace:
        raise ValueError("namespace must be nonempty")
    key = {
        "schema_version": SEED_SCHEMA_VERSION,
        "namespace": str(namespace),
        "components": [str(component) for component in components],
    }
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    seed = int.from_bytes(digest[:16], byteorder="big", signed=False)
    record = {
        **key,
        "canonical_key": canonical.decode("utf-8"),
        "sha256": digest.hex(),
        "seed_uint128_decimal": str(seed),
        "bit_generator": BIT_GENERATOR,
    }
    return seed, record


def make_generator(namespace: str, *components: object) -> Tuple[np.random.Generator, Dict[str, Any]]:
    seed, record = derive_seed(namespace, *components)
    return np.random.Generator(np.random.PCG64DXSM(seed)), record


def generator_from_state(state: Dict[str, Any]) -> np.random.Generator:
    """Restore a NumPy generator from its recorded bit-generator state."""

    name = state.get("bit_generator")
    if name != BIT_GENERATOR:
        raise ValueError(f"Unsupported bit generator {name!r}; expected {BIT_GENERATOR}")
    bit_generator = np.random.PCG64DXSM()
    bit_generator.state = state
    return np.random.Generator(bit_generator)

