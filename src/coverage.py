"""World-aligned voxelisation and deterministic occupied-voxel metrics."""

import itertools
import math

import numpy as np


def translate_aligned_indices(
    indices, source_origin, source_voxel_size, target_origin, target_voxel_size
):
    """Translate discrete source-grid indices onto an aligned target grid."""
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape (N, 3)")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must use an integer dtype")

    source_origin = np.asarray(source_origin, dtype=np.float64)
    target_origin = np.asarray(target_origin, dtype=np.float64)
    if source_origin.shape != (3,) or target_origin.shape != (3,):
        raise ValueError("source and target origins must have shape (3,)")
    if not np.all(np.isfinite(source_origin)) or not np.all(np.isfinite(target_origin)):
        raise ValueError("source and target origins must be finite")

    source_voxel_size = float(source_voxel_size)
    target_voxel_size = float(target_voxel_size)
    if (
        not math.isfinite(source_voxel_size)
        or not math.isfinite(target_voxel_size)
        or source_voxel_size <= 0
        or target_voxel_size <= 0
    ):
        raise ValueError("source and target voxel sizes must be positive and finite")
    if not math.isclose(
        source_voxel_size, target_voxel_size, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "Bonxai map resolution must equal the evaluation voxel size for "
            "exact integer alignment"
        )

    offset_float = (source_origin - target_origin) / target_voxel_size
    offset = np.rint(offset_float).astype(np.int64)
    if not np.allclose(offset_float, offset, rtol=0.0, atol=1e-9):
        raise ValueError(
            "Bonxai map origin is not aligned to the evaluation voxel grid"
        )

    source_indices = indices.astype(np.int64, copy=False)
    if len(np.unique(source_indices, axis=0)) != len(source_indices):
        raise ValueError("Bonxai map contains duplicate map_index values")
    return source_indices + offset


def aligned_index_keys(
    indices, source_origin, source_voxel_size, target_origin, target_voxel_size
):
    """Return unique keys translated from a compatible discrete source grid."""
    translated = translate_aligned_indices(
        indices,
        source_origin,
        source_voxel_size,
        target_origin,
        target_voxel_size,
    )
    return {tuple(row) for row in translated}


def voxel_keys(points, origin, voxel_size):
    """Return unique integer voxel tuples for an Nx3 point array."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    keys = np.floor((points - np.asarray(origin, dtype=np.float64)) / voxel_size)
    return {tuple(row) for row in keys.astype(np.int64)}


def neighbour_offsets(voxel_size, tolerance):
    """Offsets whose voxel-centre distance is within tolerance."""
    if tolerance < 0:
        raise ValueError("match tolerance must be non-negative")
    radius = int(math.floor(tolerance / voxel_size + 1e-12))
    return tuple(
        offset
        for offset in itertools.product(range(-radius, radius + 1), repeat=3)
        if math.sqrt(sum(value * value for value in offset)) * voxel_size
        <= tolerance + 1e-12
    )


def matched_source(source, target, offsets):
    """Source keys having at least one tolerance-neighbour in target."""
    if not source or not target:
        return set()
    matched = set()
    progress_interval = 500000
    for index, key in enumerate(source, start=1):
        if any(
            (key[0] + dx, key[1] + dy, key[2] + dz) in target
            for dx, dy, dz in offsets
        ):
            matched.add(key)
        if len(source) >= progress_interval and index % progress_interval == 0:
            print(
                f"Matching voxels: {index}/{len(source)}",
                flush=True,
            )
    return matched


def occupied_voxel_metrics(ground_truth, occupied, offsets):
    matched_gt = matched_source(ground_truth, occupied, offsets)
    matched_occupied = matched_source(occupied, ground_truth, offsets)
    recall = len(matched_gt) / len(ground_truth) if ground_truth else 0.0
    precision = len(matched_occupied) / len(occupied) if occupied else 0.0
    f1 = 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {
        "ground_truth_voxels": len(ground_truth),
        "occupied_voxels": len(occupied),
        "matched_ground_truth_voxels": len(matched_gt),
        "matched_occupied_voxels": len(matched_occupied),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def keys_in_bounds(keys, origin, voxel_size, bounds_min, bounds_max):
    """Select keys whose voxel centres lie in the half-open region bounds."""
    if not keys:
        return set()
    origin = np.asarray(origin, dtype=np.float64)
    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    array = np.asarray(tuple(keys), dtype=np.int64)
    centres = origin + (array + 0.5) * voxel_size
    mask = np.all((centres >= lower) & (centres < upper), axis=1)
    return {tuple(row) for row in array[mask]}


def evaluate_region(ground_truth, agents, offsets):
    """Evaluate one region, including union contribution and conflicts."""
    swarm_occupied = set().union(*(item["occupied"] for item in agents))
    swarm_free = set().union(*(item["free"] for item in agents))
    swarm_known = swarm_occupied | swarm_free
    conflicts = swarm_occupied & swarm_free
    swarm = occupied_voxel_metrics(ground_truth, swarm_occupied, offsets)
    swarm["known_voxels"] = len(swarm_known)
    swarm["classification_conflicts"] = len(conflicts)
    swarm["classification_conflict_fraction"] = (
        len(conflicts) / len(swarm_known) if swarm_known else 0.0
    )

    agent_metrics = []
    matched_all = swarm["matched_ground_truth_voxels"]
    for index, agent in enumerate(agents):
        metrics = occupied_voxel_metrics(ground_truth, agent["occupied"], offsets)
        others = set().union(
            *(other["occupied"] for other_index, other in enumerate(agents) if other_index != index)
        )
        without_count = occupied_voxel_metrics(ground_truth, others, offsets)[
            "matched_ground_truth_voxels"
        ]
        metrics.update(
            {
                "agent_id": agent["agent_id"],
                "map": agent["map"],
                "known_voxels": len(agent["occupied"] | agent["free"]),
                "free_voxels": len(agent["free"]),
                "unique_contribution_voxels": matched_all - without_count,
                "overlap_occupied_voxels": len(agent["occupied"] & others),
            }
        )
        agent_metrics.append(metrics)

    best = max((item["recall"] for item in agent_metrics), default=0.0)
    swarm["best_agent_recall"] = best
    swarm["improvement_over_best_agent"] = swarm["recall"] - best
    return swarm, agent_metrics
