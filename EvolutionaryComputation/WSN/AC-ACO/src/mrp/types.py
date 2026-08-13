"""Small MRP contracts that do not replace existing simulator objects."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ClusterFormationResult:
    """MRP_PHASE_I result for one event area, before any routing exists."""

    cluster_head: int
    members: tuple[int, ...]
    scores: Mapping[int, float]
    neighbor_counts: Mapping[int, int]

    @classmethod
    def create(
        cls,
        cluster_head: int,
        members: tuple[int, ...],
        scores: dict[int, float],
        neighbor_counts: dict[int, int],
    ) -> "ClusterFormationResult":
        """Copy diagnostics so later caller mutation cannot change this result."""

        return cls(
            cluster_head=cluster_head,
            members=members,
            scores=MappingProxyType(scores.copy()),
            neighbor_counts=MappingProxyType(neighbor_counts.copy()),
        )
