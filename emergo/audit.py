"""Append-only audit log for Emergo operations.

INV-7: Observable + Fail-Closed.
Every CE execution attempt (success or failure) produces an AuditRecord.
Records are written before state is committed and cannot be modified afterward.

AuditLog is a thin wrapper used inside the Lux bridge; this module provides
helper types for tests and inspection.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class AuditRecord:
    """Immutable record of a single CE execution attempt."""

    audit_id: str
    timestamp: float            # unix epoch
    ce_type: str
    agent_ids: tuple
    success: bool
    resource_deducted: float    # 0.0 if not applicable
    details: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict) -> AuditRecord:
        return cls(
            audit_id=d["audit_id"],
            timestamp=d["timestamp"],
            ce_type=d["ce_type"],
            agent_ids=tuple(d["agent_ids"]),
            success=d["success"],
            resource_deducted=float(d.get("resource_deducted", 0.0)),
            details=dict(d.get("details", {})),
        )


def typed_log(raw_records: Sequence[dict]) -> List[AuditRecord]:
    """Convert a raw audit log (list of dicts from SimulatedLuxBridge) into AuditRecords."""
    return [AuditRecord.from_dict(r) for r in raw_records]
