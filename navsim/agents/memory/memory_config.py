#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory agent configuration for navsim.
Replaces the external precomputation.config dependency.
"""

import os


def get_env_or_raise(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise KeyError(
            f"\n\n[Missing env var] '{key}' is not set."
            f"\nRun: export {key}=\"/path/to/...\"\n"
        )
    return value


# --- Environment variables ---
NUPLAN_MAPS_ROOT = get_env_or_raise("NUPLAN_MAPS_ROOT")
NUPLAN_GSV_EMBEDDING_ROOT = get_env_or_raise("NUPLAN_GSV_EMBEDDING_ROOT")

# --- Map ---
NUPLAN_MAP_DIRECTORY = NUPLAN_MAPS_ROOT

MAP_NAMES_NUPLAN = [
    "sg-one-north",
    "us-ma-boston",
    "us-nv-las-vegas-strip",
    "us-pa-pittsburgh-hazelwood",
]

# --- Embeddings ---
NUPLAN_MEMORY_EMBEDDING_DIR = NUPLAN_GSV_EMBEDDING_ROOT

NUPLAN_DINOV2_EMBEDDING_PICKLEFILE = os.path.join(
    NUPLAN_MEMORY_EMBEDDING_DIR, "dinov2", "dino_embedding_all.pkl"
)
NUPLAN_SEGFORMER_EMBEDDING_PICKLEFILE = os.path.join(
    NUPLAN_MEMORY_EMBEDDING_DIR, "segformer", "segformer_embedding_all.pkl"
)
NUPLAN_SIGLIP2_EMBEDDING_PICKLEFILE = os.path.join(
    NUPLAN_MEMORY_EMBEDDING_DIR, "siglip2", "siglip2_embedding_all.pkl"
)

# --- Parameters ---
DISTANCE_BIN_SIZE = 5
MEMORY_NODE_NUM = 20
