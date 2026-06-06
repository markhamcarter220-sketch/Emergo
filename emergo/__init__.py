"""Emergo Kernel — self-referential graph dynamical system.

Public surface:
    Graph, CoordinationEvent, PhiMap, Authority, Errors  (types)
    Lux                                                   (authorization oracle)
    ce_execute, error_computation, authority_update, phi_update  (four operations)
    emergo_kernel, make_initial_phi, make_initial_authority       (kernel + factory)
"""

from emergo.types import (
    Authority,
    CoordinationEvent,
    Errors,
    Graph,
    PhiMap,
    State,
)
from emergo.lux import Lux
from emergo.ce_execution import ce_execute
from emergo.error_computation import error_computation
from emergo.authority_update import authority_update
from emergo.phi_update import phi_update
from emergo.kernel import emergo_kernel, make_initial_phi, make_initial_authority

__all__ = [
    "Graph",
    "CoordinationEvent",
    "PhiMap",
    "Authority",
    "Errors",
    "State",
    "Lux",
    "ce_execute",
    "error_computation",
    "authority_update",
    "phi_update",
    "emergo_kernel",
    "make_initial_phi",
    "make_initial_authority",
]
