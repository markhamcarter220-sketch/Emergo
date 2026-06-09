"""State persistence — save and load the full Emergo kernel state.

Provides two backends:
  - pickle (default): fast, lossless, not human-readable
  - json: human-readable approximation (numpy arrays serialized as lists)

Usage::

    from emergo.persistence import save_state, load_state

    state = (G, phi, A, E_history)
    save_state(state, "run_checkpoint.pkl")

    state2 = load_state("run_checkpoint.pkl")
    G2, phi2, A2, E2 = state2
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np

from emergo.types import Authority, Errors, Graph, PhiMap, State


def save_state(
    state: State,
    path: str | Path,
    *,
    format: str = "pickle",
) -> Path:
    """Persist a kernel state tuple (G, phi, A, E) to disk.

    Args:
        state:   The (Graph, PhiMap, Authority, list[Errors]) tuple to save.
        path:    Destination file path. Extension is added if missing.
        format:  "pickle" (default) or "json".

    Returns:
        The resolved path where the file was written.
    """
    path = Path(path)
    if format == "pickle":
        if not path.suffix:
            path = path.with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif format == "json":
        if not path.suffix:
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _state_to_dict(state)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unknown format {format!r}. Use 'pickle' or 'json'.")
    return path


def load_state(path: str | Path) -> State:
    """Load a kernel state tuple from disk.

    Automatically detects format from file extension (.pkl → pickle, .json → json).

    Args:
        path:  Path to the saved state file.

    Returns:
        The (Graph, PhiMap, Authority, list[Errors]) state tuple.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is not recognised.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)  # type: ignore[no-any-return]
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        return _dict_to_state(data)
    else:
        raise ValueError(f"Unrecognised extension {path.suffix!r}. Use .pkl or .json.")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _state_to_dict(state: State) -> dict:
    G, phi, A, E = state
    return {
        "graph": {
            "agent_ids": list(G.agent_ids),
            "adjacency": G.adjacency.tolist(),
            "capabilities": G.capabilities.tolist(),
        },
        "phi": {
            "W_phi": phi.W_phi.tolist(),
            "b_phi": phi.b_phi.tolist(),
            "W_F": phi.W_F.tolist(),
            "b_F": phi.b_F.tolist(),
            "d_latent": phi.d_latent,
            "d_features": phi.d_features,
            "d_ce": phi.d_ce,
        },
        "authority": {
            "scores": dict(A.scores),
            "baseline": A.baseline,
        },
        "errors": [{"per_agent": dict(e.per_agent)} for e in E],
    }


def _dict_to_state(data: dict) -> State:
    gd = data["graph"]
    G = Graph(
        agent_ids=tuple(gd["agent_ids"]),
        adjacency=np.array(gd["adjacency"], dtype=float),
        capabilities=np.array(gd["capabilities"], dtype=float),
    )
    pd = data["phi"]
    phi = PhiMap(
        W_phi=np.array(pd["W_phi"], dtype=float),
        b_phi=np.array(pd["b_phi"], dtype=float),
        W_F=np.array(pd["W_F"], dtype=float),
        b_F=np.array(pd["b_F"], dtype=float),
        d_latent=pd["d_latent"],
        d_features=pd["d_features"],
        d_ce=pd["d_ce"],
    )
    ad = data["authority"]
    A = Authority(scores=dict(ad["scores"]), baseline=float(ad["baseline"]))
    E = [Errors(per_agent=dict(e["per_agent"])) for e in data.get("errors", [])]
    return G, phi, A, E
