"""Kernel state serialization / deserialization.

Saves and loads the full Emergo kernel State:
    State = tuple[Graph, PhiMap, Authority, list[Errors]]

Format
------
Checkpoint directory contains two files:
  emergo_state.npz   — all numpy arrays (compressed)
  emergo_meta.json   — structural metadata (agent IDs, dimensions, error history)

The .json file is human-readable and can be inspected without loading Python.

Usage
-----
    from emergo.serde import save_checkpoint, load_checkpoint

    save_checkpoint(state, Path("./checkpoints/run_001"))
    state, meta = load_checkpoint(Path("./checkpoints/run_001"))

Invariants
----------
- Round-trip: load(save(state)) produces a State numerically identical to
  the original (np.allclose on all arrays, exact equality on scalars/strings).
- Atomic write: arrays written to .npz before .json; if .json write fails,
  the checkpoint is incomplete and load_checkpoint will raise.
- Version tag: meta.json includes a 'serde_version' field for future
  compatibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from emergo.types import Authority, Errors, Graph, PhiMap, State

logger = logging.getLogger(__name__)

_SERDE_VERSION = "1.0"


def save_checkpoint(
    state: State,
    directory: Path | str,
    metadata: dict | None = None,
) -> Path:
    """Save kernel state to `directory`. Creates directory if needed.

    Args:
        state:      Kernel state tuple (Graph, PhiMap, Authority, list[Errors]).
        directory:  Target directory path. Created if absent.
        metadata:   Optional caller-supplied metadata (e.g. iteration number,
                    convergence reason). Stored in emergo_meta.json.

    Returns:
        Path to the checkpoint directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    G, phi, A, error_history = state

    npz_path = directory / "emergo_state.npz"
    np.savez_compressed(
        npz_path,
        graph_adjacency=G.adjacency,
        graph_capabilities=G.capabilities,
        phi_W_phi=phi.W_phi,
        phi_b_phi=phi.b_phi,
        phi_W_F=phi.W_F,
        phi_b_F=phi.b_F,
    )
    logger.debug("save_checkpoint: arrays written to %s", npz_path)

    error_history_data = [
        {"per_agent": eh.per_agent, "proposer_id": eh.proposer_id}
        for eh in error_history
    ]

    meta: dict = {
        "serde_version": _SERDE_VERSION,
        "graph": {"agent_ids": list(G.agent_ids)},
        "phi": {
            "d_latent": phi.d_latent,
            "d_features": phi.d_features,
            "d_ce": phi.d_ce,
        },
        "authority": {"scores": A.scores, "baseline": A.baseline},
        "error_history": error_history_data,
    }
    if metadata:
        meta["caller_metadata"] = metadata

    json_path = directory / "emergo_meta.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.debug("save_checkpoint: metadata written to %s", json_path)

    return directory


def load_checkpoint(directory: Path | str) -> tuple[State, dict]:
    """Load kernel state from a checkpoint directory.

    Args:
        directory: Path to checkpoint directory created by save_checkpoint.

    Returns:
        (state, caller_metadata) where state is a valid kernel State tuple
        and caller_metadata is whatever was passed to save_checkpoint (empty
        dict if not provided).

    Raises:
        FileNotFoundError: if either .npz or .json is missing.
        ValueError: if the serde_version is incompatible.
    """
    directory = Path(directory)
    npz_path = directory / "emergo_state.npz"
    json_path = directory / "emergo_meta.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Checkpoint arrays not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)

    version = meta.get("serde_version", "unknown")
    if version != _SERDE_VERSION:
        raise ValueError(
            f"Checkpoint serde_version {version!r} != current {_SERDE_VERSION!r}. "
            "Re-save the checkpoint with the current version of Emergo."
        )

    arrays = np.load(npz_path)

    agent_ids = tuple(meta["graph"]["agent_ids"])
    G = Graph(
        agent_ids=agent_ids,
        adjacency=arrays["graph_adjacency"],
        capabilities=arrays["graph_capabilities"],
    )

    phi_meta = meta["phi"]
    phi = PhiMap(
        W_phi=arrays["phi_W_phi"],
        b_phi=arrays["phi_b_phi"],
        W_F=arrays["phi_W_F"],
        b_F=arrays["phi_b_F"],
        d_latent=phi_meta["d_latent"],
        d_features=phi_meta["d_features"],
        d_ce=phi_meta["d_ce"],
    )

    auth_meta = meta["authority"]
    A = Authority(
        scores={k: float(v) for k, v in auth_meta["scores"].items()},
        baseline=float(auth_meta["baseline"]),
    )

    error_history: list[Errors] = [
        Errors(
            per_agent={k: float(v) for k, v in eh["per_agent"].items()},
            proposer_id=eh.get("proposer_id"),
        )
        for eh in meta.get("error_history", [])
    ]

    state: State = (G, phi, A, error_history)
    caller_metadata: dict = meta.get("caller_metadata", {})

    logger.debug(
        "load_checkpoint: loaded state with %d agents, %d error history entries",
        len(agent_ids),
        len(error_history),
    )

    return state, caller_metadata
