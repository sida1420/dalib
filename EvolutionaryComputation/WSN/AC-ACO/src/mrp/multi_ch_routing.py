"""Common MRP Phase-II/III engine for one or more independently routed CHs."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .phase2.routing import (
    _control_ledger,
    _run_validated_phase2_discovery,
    _validate_orchestration_inputs,
    _validated_discovery_state,
    discover_mrp_phase2_routes,
)
from .phase2.control_energy import combine_ant_control_energy, zero_ant_control_energy
from .phase3.path_selection import select_phase3_route
from .runner_lifecycle import copy_pheromone_state, pheromone_for_discovery
from .runner_types import MRPCHRoutingResult, PheromoneLifecycle, PureMRPParameters, PureMRPState


@dataclass(frozen=True)
class MRPMultiCHRoutingResult:
    """Selected route and pheromone snapshots for every required source CH."""

    ch_routing_results: tuple[MRPCHRoutingResult, ...]
    selected_routes: Mapping[int, tuple[int, ...]]
    pheromone_before: Mapping[int, Mapping[int, float]] | None
    pheromone_after: Mapping[int, Mapping[int, float]] | None
    control_energy: object | None = None


class MRPRoutingFailure(ValueError):
    """A classified failure with partial per-CH diagnostics and no energy commit."""

    def __init__(
        self, stage, cluster_head, classification, reason, partial_results=(),
        phase2_result=None, pheromone_before=None, pheromone_after=None,
        control_energy=None,
    ):
        self.stage = stage
        self.cluster_head = cluster_head
        self.classification = classification
        self.reason = str(reason)
        self.partial_results = tuple(partial_results)
        self.phase2_result = phase2_result
        self.pheromone_before = copy_pheromone_state(pheromone_before)
        self.pheromone_after = copy_pheromone_state(pheromone_after)
        self.control_energy = control_energy
        super().__init__(f"{classification}: CH {cluster_head}: {self.reason}")


def route_cluster_heads_with_mrp(
    cluster_heads: Sequence[int], live_nodes: Collection[int], state: PureMRPState,
    dist_matrix, base_dists, hop_counts: Mapping[int, int], parameters: PureMRPParameters,
    rng: object, pheromone_lifecycle: PheromoneLifecycle, *,
    phase2_discovery=discover_mrp_phase2_routes, phase3_selector=select_phase3_route,
) -> MRPMultiCHRoutingResult:
    """Run unchanged Phase II and III independently for every required CH."""

    heads = tuple(cluster_heads)
    live = set(live_nodes)
    if not heads or len(set(heads)) != len(heads) or any(head not in live for head in heads):
        raise ValueError("MRP routing requires distinct live source CHs")
    try:
        pheromone_before = pheromone_for_discovery(
            state, pheromone_lifecycle, dist_matrix, parameters.communication_radius,
            parameters.config,
        )
    except Exception as error:
        raise MRPRoutingFailure("phase2", heads[0], "MRP_DOMAIN_FAILURE", error) from error

    routed = []
    use_validated_hot_path = phase2_discovery is discover_mrp_phase2_routes
    if use_validated_hot_path:
        try:
            _validate_orchestration_inputs(parameters.num_sants, parameters.c, parameters.c1)
            shared_pheromone = _validated_discovery_state(
                state.live_nodes, state.residual_e, dist_matrix, base_dists,
                parameters.communication_radius, parameters.heuristic_bounds,
                parameters.config, parameters.c0,
                parameters.e_elec, parameters.free_space_coeff,
                parameters.multipath_coeff, parameters.d0, parameters.bit_count,
                pheromone_before,
            )
        except Exception as error:
            raise MRPRoutingFailure(
                "phase2", heads[0], "MRP_DOMAIN_FAILURE", error,
                pheromone_before=pheromone_before, pheromone_after=pheromone_before,
            ) from error
    else:
        shared_pheromone = pheromone_before
    for cluster_head in heads:
        active_control_ledger = None
        accounting_options = {}
        try:
            if use_validated_hot_path:
                active_control_ledger = _control_ledger(
                    parameters.charge_ant_energy,
                    parameters.ant_control_packet_bits,
                    len(state.residual_e),
                    dist_matrix,
                    base_dists,
                    parameters.e_elec,
                    parameters.free_space_coeff,
                    parameters.multipath_coeff,
                    parameters.d0,
                )
                accounting_options = {
                    "charge_ant_energy": parameters.charge_ant_energy,
                    "ant_control_packet_bits": parameters.ant_control_packet_bits,
                    "control_ledger": active_control_ledger,
                }
        except Exception as error:
            raise MRPRoutingFailure(
                "phase2", cluster_head, "MRP_DOMAIN_FAILURE", error, routed,
                pheromone_before=pheromone_before, pheromone_after=shared_pheromone,
                control_energy=_aggregate_control_energy(len(state.residual_e), routed),
            ) from error
        try:
            runner = (
                _run_validated_phase2_discovery
                if use_validated_hot_path else phase2_discovery
            )
            phase2 = runner(
                cluster_head, state.live_nodes, state.residual_e, dist_matrix, base_dists,
                hop_counts, parameters.communication_radius, parameters.heuristic_bounds,
                parameters.config, parameters.lambda_coefficient, parameters.ttl,
                parameters.num_sants, rng, parameters.c0, parameters.e_elec,
                parameters.free_space_coeff, parameters.multipath_coeff, parameters.d0,
                parameters.bit_count, parameters.c, parameters.c1, shared_pheromone,
                **accounting_options,
            )
        except Exception as error:
            raise MRPRoutingFailure(
                "phase2", cluster_head, "MRP_DOMAIN_FAILURE", error, routed,
                pheromone_before=pheromone_before, pheromone_after=shared_pheromone,
                control_energy=_aggregate_control_energy(
                    len(state.residual_e), routed,
                    control_energy=(
                        None if active_control_ledger is None
                        else active_control_ledger.snapshot()
                    ),
                ),
            ) from error
        shared_pheromone = phase2.final_pheromone_state
        if phase2.status != "completed":
            raise MRPRoutingFailure(
                "phase2", cluster_head, "MRP_DOMAIN_FAILURE", phase2.status, routed,
                phase2, pheromone_before, shared_pheromone,
                _aggregate_control_energy(len(state.residual_e), routed, phase2),
            )
        try:
            phase3 = phase3_selector(phase2, parameters.config, rng)
        except Exception as error:
            classification = (
                "PHYSICAL_CONNECTIVITY_EXHAUSTED"
                if cluster_head not in hop_counts else "MRP_SEARCH_FAILURE"
            )
            raise MRPRoutingFailure(
                "phase3", cluster_head, classification, error, routed, phase2,
                pheromone_before, shared_pheromone,
                _aggregate_control_energy(len(state.residual_e), routed, phase2),
            ) from error
        routed.append(MRPCHRoutingResult(cluster_head, phase2, phase3))

    selected = {item.cluster_head: tuple(item.phase3_result.selected_route) for item in routed}
    return MRPMultiCHRoutingResult(
        tuple(routed), MappingProxyType(selected), copy_pheromone_state(pheromone_before),
        copy_pheromone_state(shared_pheromone),
        _aggregate_control_energy(len(state.residual_e), routed),
    )


def _aggregate_control_energy(
    node_count, routed, phase2=None, control_energy=None,
):
    results = [item.phase2_result for item in routed]
    if phase2 is not None:
        results.append(phase2)
    ledgers = [
        ledger for item in results
        if (ledger := getattr(item, "control_energy", None)) is not None
    ]
    if control_energy is not None:
        ledgers.append(control_energy)
    return (
        combine_ant_control_energy(*ledgers)
        if ledgers else zero_ant_control_energy(node_count)
    )
