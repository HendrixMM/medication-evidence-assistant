"""
Shared model normalization utilities for embedding and rerank models.

Provides helpers to convert between short and full model identifiers and
handles common aliasing (e.g., underscore variants).
"""
from __future__ import annotations

# Known short -> full mappings
EMBEDDING_MODEL_MAP: dict[str, str] = {
    "nv-embedqa-e5-v5": "nvidia/nv-embedqa-e5-v5",
    "nv-embedqa-mistral7b-v2": "nvidia/nv-embedqa-mistral7b-v2",
    "snowflake-arctic-embed-l": "Snowflake/snowflake-arctic-embed-l",
}

RERANK_MODEL_MAP: dict[str, str] = {
    "nv-rerankqa-mistral4b-v3": "nvidia/nv-rerankqa-mistral4b-v3",
    # Accept underscore variant and map to full meta name
    "llama-3_2-nemoretriever-500m-rerank-v2": "meta/llama-3_2-nemoretriever-500m-rerank-v2",
}


def normalize_model(model: str | None, prefer_full_name: bool = True) -> str | None:
    """Normalize model names across embedding and rerank families.

    - prefer_full_name=True maps known short names to their fully qualified names.
    - prefer_full_name=False maps full names back to short when known.
    - Underscore variant for llama rerank is preserved but namespaced.
    """
    if not model:
        return model

    # Direct mapping for embedding and rerank
    if prefer_full_name:
        if model in EMBEDDING_MODEL_MAP:
            return EMBEDDING_MODEL_MAP[model]
        if model in RERANK_MODEL_MAP:
            return RERANK_MODEL_MAP[model]
        # Ensure namespace for common NV prefixes when already short
        if model.startswith("nv-") and "embed" in model and not model.startswith("nvidia/"):
            return f"nvidia/{model}"
        if model.startswith("nv-") and "rerank" in model and not model.startswith("nvidia/"):
            return f"nvidia/{model}"
        # Llama underscore variant without namespace
        if model.startswith("llama-3_2") and "rerank" in model and not model.startswith("meta/"):
            return f"meta/{model}"
        return model
    else:
        # Reverse map for embedding
        rev_embed = {v: k for k, v in EMBEDDING_MODEL_MAP.items()}
        if model in rev_embed:
            return rev_embed[model]
        # Reverse map for rerank
        rev_rerank = {v: k for k, v in RERANK_MODEL_MAP.items()}
        if model in rev_rerank:
            return rev_rerank[model]
        # Drop namespace as best-effort short form
        if "/" in model:
            return model.split("/")[-1]
        return model
