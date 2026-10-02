#!/usr/bin/env python3
"""Evaluate saved Bonxai maps against ground-truth reference voxels."""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from coverage import (
    evaluate_region, keys_in_bounds, matched_source, neighbour_offsets,
    occupied_voxel_metrics, translate_aligned_indices,
)
from ground_truth import load_voxels
from report import show_interactive_3d, write_reports


def _required(mapping, key, context):
    if key not in mapping:
        raise ValueError(f"{context} is missing required field '{key}'")
    return mapping[key]


def _bounds(value, context):
    lower = np.asarray(_required(value, "min", context), dtype=np.float64)
    upper = np.asarray(_required(value, "max", context), dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError(f"{context} must have valid 3D min/max bounds")
    return lower, upper


def _visualisation_enabled(evaluation, override):
    configured = evaluation.get("vis", False)
    if not isinstance(configured, bool):
        raise ValueError("evaluation.vis must be true or false")
    return configured if override is None else bool(override)


def _agent_id_from_map_path(map_path):
    match = re.fullmatch(r"agents?(\d+)(?:_(?:final|map))?", map_path.stem)
    return match.group(1) if match else map_path.stem


def _literal_config_value(path, name):
    """Read one literal assignment from a frozen legacy config without executing it."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path} is missing literal assignment {name}")


def _mission_context(manifest, base, map_paths):
    mission_config = manifest.get("mission_config")
    if not isinstance(mission_config, dict):
        return None
    teams = mission_config.get("teams")
    if not isinstance(teams, list) or not teams:
        raise ValueError("mission_config.teams must be a non-empty list")

    dynamic_enabled = mission_config.get("dynamic_outlines_enabled")
    dynamic_ids = mission_config.get("dynamic_grid_ids")
    if dynamic_enabled is None or dynamic_ids is None:
        frozen_config = base / str(mission_config.get("source", "v_configs.py"))
        dynamic_enabled = bool(_literal_config_value(
            frozen_config, "FRONTIER_DYNAMIC_OUTLINE_ENABLED"
        ))
        dynamic_ids = _literal_config_value(
            frozen_config, "FRONTIER_DYNAMIC_OUTLINE_GRIDS"
        )
    dynamic_ids = [int(value) for value in dynamic_ids] if dynamic_enabled else []

    outcome_paths = ((manifest.get("mission_results") or {}).get("grid_outcomes") or {})
    if not isinstance(outcome_paths, dict):
        raise ValueError("mission_results.grid_outcomes must be a mapping")
    desired_agents = sorted({
        int(agent_id)
        for team in teams
        for agent_id in _required(team, "agent_ids", "mission team")
    })
    candidates = {int(agent_id): (base / value).resolve()
                  for agent_id, value in outcome_paths.items()}
    artifact_dirs = {path.parent for path in map_paths}
    for agent_id in desired_agents:
        if agent_id in candidates:
            continue
        for artifact_dir in artifact_dirs:
            candidate = artifact_dir / f"agent{agent_id:03d}_grid_outcomes.json"
            if candidate.is_file():
                candidates[agent_id] = candidate
                break

    outcomes = {}
    for agent_id, path in sorted(candidates.items()):
        if not path.is_file():
            raise FileNotFoundError(f"grid outcome file does not exist: {path}")
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        raw_outcomes = _required(payload, "grid_outcomes", str(path))
        if not isinstance(raw_outcomes, dict):
            raise ValueError(f"{path}: grid_outcomes must be a mapping")
        agent_outcomes = {}
        for grid_id, value in raw_outcomes.items():
            status = _required(value, "status", f"{path} grid {grid_id}")
            if status not in {"local_complete", "incomplete"}:
                raise ValueError(f"{path}: unsupported grid status {status!r}")
            agent_outcomes[str(int(grid_id))] = {
                "status": status,
                "reason": str(value.get("reason", "")),
            }
        outcomes[str(agent_id)] = agent_outcomes

    return {
        "teams": teams,
        "dynamic_outlines_enabled": bool(dynamic_enabled),
        "dynamic_grid_ids": dynamic_ids,
        "grid_outcomes": outcomes,
        "team_grid_metrics": {},
        "second_floor": mission_config.get("second_floor"),
    }


def evaluate(manifest_path, output_dir, voxel_override=None, tolerance_override=None,
             region_name=None, plots=True, rebuild_cache=False, vis_override=None):
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a YAML mapping")
    base = manifest_path.parent
    run_id = str(_required(manifest, "run_id", "manifest"))
    gt_path = (base / _required(manifest, "ground_truth", "manifest")).resolve()
    if not gt_path.is_file():
        raise FileNotFoundError(f"ground truth does not exist: {gt_path}")
    map_paths = [(base / item).resolve() for item in _required(manifest, "agent_maps", "manifest")]
    missing = [str(path) for path in map_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("required agent map(s) missing: " + ", ".join(missing))
    if not map_paths:
        raise ValueError("manifest agent_maps must not be empty")
    mission = _mission_context(manifest, base, map_paths)

    evaluation = _required(manifest, "evaluation", "manifest")
    visualise = _visualisation_enabled(evaluation, vis_override)
    frame = str(_required(evaluation, "frame_id", "evaluation"))
    voxel_size = float(voxel_override if voxel_override is not None else
                       _required(evaluation, "voxel_size_m", "evaluation"))
    tolerance = float(tolerance_override if tolerance_override is not None else
                      _required(evaluation, "match_tolerance_m", "evaluation"))
    if voxel_size <= 0:
        raise ValueError("voxel size must be positive")
    primary_min, primary_max = _bounds(
        _required(evaluation, "bounds", "evaluation"), "evaluation.bounds"
    )
    origin = primary_min.copy()
    # Load every explicitly evaluated region, retaining the primary report bounds.
    all_bounds = [(primary_min, primary_max)] + [
        _bounds(value, f"evaluation.regions.{name}")
        for name, value in (evaluation.get("regions") or {}).items()
    ]
    load_min = np.min([lower for lower, _ in all_bounds], axis=0)
    load_max = np.max([upper for _, upper in all_bounds], axis=0)

    transform = _required(manifest, "ground_truth_transform", "manifest")
    translation = _required(transform, "translation_xyz", "ground_truth_transform")
    quaternion = _required(transform, "quaternion_xyzw", "ground_truth_transform")
    gt = load_voxels(
        gt_path, origin, voxel_size, load_min, load_max, translation, quaternion
    )

    try:
        from detect_stairs.bonxai_map import (
            STATE_FREE, STATE_OCCUPIED, load_map,
        )
    except ImportError as exc:
        raise RuntimeError(
            "detect_stairs is required; source the ROS workspace before evaluation"
        ) from exc

    agents = []
    map_resolution = None
    for map_index, map_path in enumerate(map_paths):
        cache = cache_dir / f"{map_index:03d}_{map_path.name}.npz"
        if rebuild_cache and cache.exists():
            cache.unlink()
        metadata, indices, _positions, states = load_map(map_path, cache_file=cache)
        if metadata.get("frame_id") != frame:
            raise ValueError(
                f"{map_path}: frame {metadata.get('frame_id')!r} does not match {frame!r}"
            )
        resolution = float(_required(metadata, "resolution", str(map_path)))
        if map_resolution is None:
            map_resolution = resolution
        elif not np.isclose(map_resolution, resolution, rtol=0.0, atol=1e-9):
            raise ValueError("agent Bonxai map resolutions differ")
        map_origin = np.asarray(
            _required(metadata, "global_origin", str(map_path)), dtype=np.float64
        )
        try:
            translated_indices = translate_aligned_indices(
                indices,
                map_origin,
                resolution,
                origin,
                voxel_size,
            )
        except ValueError as exc:
            raise ValueError(f"{map_path}: {exc}") from exc
        occupied = {
            tuple(row) for row in translated_indices[states == STATE_OCCUPIED]
        }
        free = {tuple(row) for row in translated_indices[states == STATE_FREE]}
        occupied = keys_in_bounds(
            occupied, origin, voxel_size, load_min, load_max
        )
        free = keys_in_bounds(free, origin, voxel_size, load_min, load_max)
        agents.append({
            "agent_id": _agent_id_from_map_path(map_path),
            "map": str(map_path),
            "occupied": occupied,
            "free": free,
        })

    configured_regions = {"whole_environment": [(primary_min, primary_max)]}
    for name, value in (evaluation.get("regions") or {}).items():
        configured_regions[str(name)] = [
            _bounds(value, f"evaluation.regions.{name}")
        ]
    for name, members in (evaluation.get("region_unions") or {}).items():
        if not isinstance(members, list) or not members:
            raise ValueError(f"evaluation.region_unions.{name} must be a non-empty list")
        missing_members = [
            str(member) for member in members if str(member) not in configured_regions
        ]
        if missing_members:
            raise ValueError(
                f"evaluation.region_unions.{name} references unknown regions: "
                + ", ".join(missing_members)
            )
        configured_regions[str(name)] = [
            bounds
            for member in members
            for bounds in configured_regions[str(member)]
        ]
    grid_regions = {
        str(name): configured_regions[str(name)]
        for name in (evaluation.get("regions") or {})
        if re.fullmatch(r"grid_\d+", str(name))
    }
    primary_region = str(evaluation.get("primary_region", "whole_environment"))
    if primary_region not in configured_regions:
        raise ValueError(f"unknown primary region {primary_region!r}")
    if region_name:
        if region_name not in configured_regions:
            raise ValueError(f"unknown region {region_name!r}")
        configured_regions = {region_name: configured_regions[region_name]}
        primary_region = region_name
    offsets = neighbour_offsets(voxel_size, tolerance)
    regions = {}
    region_agents = {}
    primary_plot = None
    additional_plots = {}
    additional_regions = evaluation.get("additional_plot_regions", [])
    for name, region_bounds in configured_regions.items():
        print(f"Evaluating region: {name}", flush=True)
        region_gt = set().union(*(
            keys_in_bounds(gt, origin, voxel_size, lower, upper)
            for lower, upper in region_bounds
        ))
        if not region_gt:
            raise ValueError(f"ground truth is empty in region {name!r}")
        filtered_agents = []
        for agent in agents:
            occupied = set().union(*(
                keys_in_bounds(agent["occupied"], origin, voxel_size, lower, upper)
                for lower, upper in region_bounds
            ))
            free = set().union(*(
                keys_in_bounds(agent["free"], origin, voxel_size, lower, upper)
                for lower, upper in region_bounds
            ))
            filtered_agents.append({
                **agent,
                "occupied": occupied,
                "free": free,
            })
        swarm, per_agent = evaluate_region(region_gt, filtered_agents, offsets)
        regions[name] = swarm
        region_agents[name] = per_agent
        if mission is not None and re.fullmatch(r"grid_\d+(?:_second_floor)?", name):
            grid_id = int(name.split("_")[1])
            for team in mission["teams"]:
                grid_order = [int(value) for value in team["frontier_grid_order"]]
                if grid_id not in grid_order:
                    continue
                team_agent_ids = {int(value) for value in team["agent_ids"]}
                team_occupied = set().union(*(
                    agent["occupied"] for agent in filtered_agents
                    if int(agent["agent_id"]) in team_agent_ids
                ))
                team_metrics = occupied_voxel_metrics(
                    region_gt, team_occupied, offsets
                )
                mission["team_grid_metrics"].setdefault(
                    str(int(team["team_id"])), {}
                )[name] = team_metrics
        if name == primary_region or name in additional_regions:
            union = set().union(*(agent["occupied"] for agent in filtered_agents))
            region_plot = {
                "ground_truth": region_gt,
                "matched_ground_truth": matched_source(
                    region_gt, union, offsets
                ),
                "agent_union": union,
                "origin": origin,
                "voxel_size": voxel_size,
                "bounds_min": (primary_min if name == primary_region else
                               np.min([lower for lower, _ in region_bounds], axis=0)),
                "bounds_max": (primary_max if name == primary_region else
                               np.max([upper for _, upper in region_bounds], axis=0)),
                "grid_regions": ({key: configured_regions[key] for key in
                    (evaluation.get("region_unions") or {}).get(name, [])}
                    if name in additional_regions else grid_regions),
            }

            if name == primary_region:
                primary_plot = region_plot
            else:
                additional_plots[name] = region_plot

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "manifest": str(manifest_path),
        "ground_truth": str(gt_path),
        "agent_maps": [str(path) for path in map_paths],
        "agents_evaluated": len(agents),
        "primary_region": primary_region,
        "evaluation": {
            "frame_id": frame,
            "voxel_size_m": voxel_size,
            "match_tolerance_m": tolerance,
            "bounds": {"min": primary_min.tolist(), "max": primary_max.tolist()},
            "bonxai_resolution_m": map_resolution,
            "vis": visualise,
        },
        "regions": regions,
    }
    if mission is not None:
        summary["mission"] = mission
    write_reports(
        output_dir,
        summary,
        region_agents,
        primary_plot if plots else None,
        interactive_backend=visualise,
        additional_plot_data=additional_plots if plots else None,
    )
    if visualise:
        print("Interactive 3D viewer opened; close its window to finish.")
        show_interactive_3d(primary_plot, run_id, primary_region)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--match-tolerance", type=float)
    parser.add_argument("--region")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    vis_group = parser.add_mutually_exclusive_group()
    vis_group.add_argument(
        "--vis",
        dest="vis",
        action="store_true",
        help="open the interactive 3D voxel viewer",
    )
    vis_group.add_argument(
        "--no-vis",
        dest="vis",
        action="store_false",
        help="disable the interactive 3D voxel viewer",
    )
    parser.set_defaults(vis=None)
    args = parser.parse_args()
    try:
        summary = evaluate(
            args.manifest,
            args.output,
            voxel_override=args.voxel_size,
            tolerance_override=args.match_tolerance,
            region_name=args.region,
            plots=not args.no_plots,
            rebuild_cache=args.rebuild_cache,
            vis_override=args.vis,
        )
    except Exception as exc:
        print(f"coverage evaluation failed: {exc}", file=sys.stderr)
        return 2
    metrics = summary["regions"][summary["primary_region"]]
    print(f"Run: {summary['run_id']}")
    print(f"Agents evaluated: {summary['agents_evaluated']}")
    print(f"Swarm occupied-voxel recall:    {100 * metrics['recall']:.2f}%")
    print(f"Swarm occupied-voxel precision: {100 * metrics['precision']:.2f}%")
    print(f"Swarm occupied-voxel F1:        {100 * metrics['f1']:.2f}%")
    print(f"Classification conflicts: {metrics['classification_conflicts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
