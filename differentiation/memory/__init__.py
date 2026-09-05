"""Agent memory: per-target, per-agent cross-run learning.

Three independent namespaces - one per sub-agent - persisted next to the
regression radar under ``reports/baselines/_memory/<host-slug>/``. Each
namespace is its own JSON file so a corrupt one cannot sink the other two.

What this is
------------
A way for a second run against the *same URL* to fetch what the Planner,
Generator and Healer already established, so the run spends its budget going
deeper (new flows, new selectors, new failures) instead of redoing the same
shallow work from zero.

What is deliberately NOT stored - see the per-module docstrings for the
full statement, repeated here because it is the single most important
invariant of this package: no DOM, no screenshots, no raw error text, no step
values, no full URLs, no credentials, no generated source. Every string is
redacted before it reaches disk.

What memory is not
-------------------
Not a verdict. A remembered selector is always re-verified live before use; a
remembered classification is evidence weighed against fresh evidence, never a
replacement for it. See :mod:`differentiation.memory.healer_memory` for the
explicit numeric bounds on how far memory may move the Healer's decision.
"""

from __future__ import annotations

from differentiation.memory.generator_memory import (
    AuthoringMemory,
    GeneratorMemory,
    SelectorMemory,
    load_generator_memory,
    record_generator_memory,
)
from differentiation.memory.healer_memory import (
    FailureMemory,
    HealerMemory,
    MEMORY_CONFIDENCE_INFLUENCE,
    load_healer_memory,
    record_healer_memory,
)
from differentiation.memory.keys import failure_signature, flow_key, selector_key
from differentiation.memory.planner_memory import (
    DEPTH_LEVELS,
    DepthLevel,
    FlowMemory,
    PlannerDirective,
    PlannerMemory,
    load_planner_memory,
    next_depth,
    record_planner_memory,
)
from differentiation.memory.store import (
    MemoryMeta,
    is_stale,
    memory_dir,
    read_json,
    site_fingerprint,
    write_atomic,
)

__all__ = [
    "AuthoringMemory",
    "GeneratorMemory",
    "SelectorMemory",
    "load_generator_memory",
    "record_generator_memory",
    "FailureMemory",
    "HealerMemory",
    "MEMORY_CONFIDENCE_INFLUENCE",
    "load_healer_memory",
    "record_healer_memory",
    "failure_signature",
    "flow_key",
    "selector_key",
    "DEPTH_LEVELS",
    "DepthLevel",
    "FlowMemory",
    "PlannerDirective",
    "PlannerMemory",
    "load_planner_memory",
    "next_depth",
    "record_planner_memory",
    "MemoryMeta",
    "is_stale",
    "memory_dir",
    "read_json",
    "site_fingerprint",
    "write_atomic",
]
