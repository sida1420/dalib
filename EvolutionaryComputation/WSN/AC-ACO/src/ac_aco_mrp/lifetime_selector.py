"""Exact branch-and-bound selection for experimental lifetime-aware Hybrid."""

from collections.abc import Collection, Mapping, Sequence
import math
from types import MappingProxyType

from .incremental_energy import IncrementalMultiFlowLedger, payload_only_energy
from .multiflow import build_hybrid_multiflow_plan
from .multiflow_energy import evaluate_hybrid_multiflow_energy
from .types import LifetimeSelectionResult


OBJECTIVE = "MAXIMIZE_MINIMUM_POST_ROUND_RESIDUAL_ENERGY"


def select_lifetime_aware_routes(
    cluster_heads: Sequence[int], members_by_head: Mapping[int, Sequence[int]],
    candidate_routes_by_head: Mapping[int, Sequence[Sequence[int]]],
    live_nodes: Collection[int], residual_e: Sequence[float], dist_matrix,
    base_dists, parameters,
) -> LifetimeSelectionResult:
    """Find the exact lexicographic optimum without materializing the product."""

    heads = tuple(cluster_heads)
    live = tuple(sorted(live_nodes))
    candidates = _canonical_candidates(heads, candidate_routes_by_head)
    if not live:
        raise ValueError("lifetime selection requires live sensors")
    ledger = IncrementalMultiFlowLedger(
        heads, members_by_head, len(residual_e), dist_matrix, base_dists, parameters,
    )
    vectors = tuple(tuple(
        payload_only_energy(route, len(residual_e), dist_matrix, base_dists, parameters)
        for route in routes
    ) for routes in candidates)
    suffix_vectors, suffix_totals, suffix_hops, suffix_products = _suffix_bounds(
        candidates, vectors, len(residual_e),
    )
    search_order = tuple(
        tuple(sorted(range(len(routes)), key=lambda index: _candidate_key(
            routes[index], vectors[depth][index], residual_e, live,
        )))
        for depth, routes in enumerate(candidates)
    )
    stats = {"visited": 0, "pruned": 0, "complete": 0}
    best = {"score": None, "routes": None}
    selected = []

    def visit(depth, hop_count):
        stats["visited"] += 1
        if best["score"] is not None and _cannot_beat(
            ledger, residual_e, live, suffix_vectors[depth], suffix_totals[depth],
            hop_count + suffix_hops[depth], best["score"],
        ):
            stats["pruned"] += suffix_products[depth]
            return
        if depth == len(heads):
            token = ledger.push_terminal_members()
            score = _score(ledger.e_m, residual_e, live, hop_count)
            route_key = tuple(selected)
            stats["complete"] += 1
            if (
                best["score"] is None or score > best["score"]
                or (score == best["score"] and route_key < best["routes"])
            ):
                best["score"], best["routes"] = score, route_key
            ledger.rollback(token)
            return
        head = heads[depth]
        for index in search_order[depth]:
            route = candidates[depth][index]
            token = ledger.push_flow(head, route)
            selected.append(route)
            visit(depth + 1, hop_count + len(route) - 1)
            selected.pop()
            ledger.rollback(token)

    visit(0, 0)
    selected_routes = dict(zip(heads, best["routes"]))
    plan = build_hybrid_multiflow_plan(
        heads, members_by_head, selected_routes, live, len(residual_e),
    )
    energy = evaluate_hybrid_multiflow_energy(plan, dist_matrix, base_dists, parameters)
    minimum_after = min(max(0.0, residual_e[i] - energy.e_m_list[i]) for i in live)
    total_hops = sum(len(route) - 1 for route in best["routes"])
    _require_same_score(best["score"], minimum_after, energy.e_sum, total_hops)
    return LifetimeSelectionResult(
        MappingProxyType(selected_routes), plan, energy, minimum_after, total_hops,
        stats["complete"], stats["visited"], stats["pruned"], suffix_products[0],
    )


def _canonical_candidates(heads, values):
    if not heads or set(values) != set(heads):
        raise ValueError("lifetime selection requires candidates for every selected CH")
    candidates = tuple(tuple(sorted({tuple(route) for route in values[head]})) for head in heads)
    if any(not routes for routes in candidates):
        raise ValueError("lifetime selection requires at least one route per selected CH")
    return candidates


def _suffix_bounds(candidates, vectors, node_count):
    count = len(candidates)
    suffix_vectors = [[0.0] * node_count for _ in range(count + 1)]
    totals, hops, products = [0.0] * (count + 1), [0] * (count + 1), [1] * (count + 1)
    for depth in range(count - 1, -1, -1):
        component_min = [min(vector[i] for vector in vectors[depth]) for i in range(node_count)]
        suffix_vectors[depth] = [
            component_min[i] + suffix_vectors[depth + 1][i] for i in range(node_count)
        ]
        totals[depth] = min(map(math.fsum, vectors[depth])) + totals[depth + 1]
        hops[depth] = min(len(route) - 1 for route in candidates[depth]) + hops[depth + 1]
        products[depth] = len(candidates[depth]) * products[depth + 1]
    return suffix_vectors, totals, hops, products


def _candidate_key(route, vector, residual_e, live):
    minimum_after = min(max(0.0, residual_e[i] - vector[i]) for i in live)
    return (-minimum_after, math.fsum(vector), len(route) - 1, route)


def _cannot_beat(ledger, residual_e, live, suffix_vector, suffix_total, lower_hops, best):
    lower = [math.nextafter(math.fsum((ledger.e_m[i], suffix_vector[i])), -math.inf) for i in live]
    upper_minimum = min(
        math.nextafter(max(0.0, residual_e[i] - value), math.inf)
        for i, value in zip(live, lower)
    )
    if upper_minimum > best[0]:
        return False
    if upper_minimum < best[0]:
        return True
    lower_total = math.nextafter(math.fsum((ledger.total_energy, suffix_total)), -math.inf)
    if lower_total < -best[1]:
        return False
    if lower_total > -best[1]:
        return True
    return lower_hops > -best[2]


def _score(e_m, residual_e, live, hops):
    return (
        min(max(0.0, residual_e[i] - e_m[i]) for i in live),
        -math.fsum(e_m), -hops,
    )


def _require_same_score(score, minimum_after, total_energy, hops):
    actual = (minimum_after, -total_energy, -hops)
    if any(not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12) for left, right in zip(score, actual)):
        raise RuntimeError("incremental lifetime score disagrees with canonical energy evaluation")
