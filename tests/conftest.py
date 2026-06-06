"""Shared test fixtures."""
import numpy as np
import pytest

from emergo.types import Authority, CoordinationEvent, Graph
from emergo.kernel import make_initial_phi, make_initial_authority
from emergo.lux import Lux


@pytest.fixture
def three_agent_graph() -> Graph:
    """Triangle graph: A→B, B→C, C→A with unit weights."""
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=float)
    caps = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
    ], dtype=float)
    return Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=caps)


@pytest.fixture
def default_authority(three_agent_graph) -> Authority:
    return make_initial_authority(three_agent_graph.agent_ids, baseline=0.5)


@pytest.fixture
def phi():
    return make_initial_phi(d_latent=8, d_features=16, d_ce=4)


@pytest.fixture
def lux():
    return Lux(min_authority=0.1)


def make_ce(event_type: str, participants: tuple, **params) -> CoordinationEvent:
    # Lists aren't hashable, so convert them to tuples for frozenset storage.
    hashable = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
    return CoordinationEvent(
        event_type=event_type,
        participants=participants,
        params=frozenset(hashable.items()),
    )
