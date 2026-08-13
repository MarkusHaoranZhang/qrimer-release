"""
Core modules for Quantum RIMER.

Public API
----------
    from core import QRIMEREngine, InferenceRequest, InferenceResult, ExecutionMode

Internal modules (implementation details, subject to change):
    core.rimer    — Classical RIMER inference engine
    core.qer      — Quantum Evidential Reasoning engine
    core.qbra     — Quantum Belief Rule Activation operator
    core.qbrb     — Quantum BRB state encoding
    core.pipeline — Hierarchical quantum inference pipeline
    core.backends — Backend abstraction for hardware deployment
    core.noise    — Noise model utilities
"""

from core.api import (
    ExecutionMode,
    InferenceRequest,
    InferenceResult,
    QRIMEREngine,
)

__all__ = [
    "QRIMEREngine",
    "InferenceRequest",
    "InferenceResult",
    "ExecutionMode",
]
