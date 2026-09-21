#!/usr/bin/env python3
"""Snapshot one artifact run and write a team-aware coverage manifest."""

import argparse
import hashlib
import importlib.util
import math
import os
import shutil
import sys
from pathlib import Path

import yaml


UTILITY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = UTILITY_ROOT.parent
DEFAULT_RUN_ROOT = UTILITY_ROOT / "runs"
DEFAULT_GROUND_TRUTH = UTILITY_ROOT / "resources" / "virtualrun_2.ply"
DEFAULT_ARTIFACT_ROOT = (
    WORKSPACE_SRC
    / "flush_search"
    / "artifacts"
)


def _load_mission_config(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"mission config does not exist: {path}")
    spec = importlib.util.spec_from_file_location("coverage_mission_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load mission config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required(mapping, key, context):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _artifact_path(artifact_run, value, context):
    path = (artifact_run / str(value)).resolve()
    if not path.is_relative_to(artifact_run):
        raise ValueError(f"{context} escapes the artifact run directory")
    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    return path


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_mission_config(artifact_run, artifact_manifest):
    inputs = _required(artifact_manifest, "inputs", "artifact manifest")
    mission_input = _required(inputs, "mission_config", "artifact manifest inputs")
    mission_config_path = _artifact_path(
        artifact_run,
        _required(mission_input, "path", "artifact mission-config input"),
        "artifact mission-config input",
    )
    expected_sha256 = _required(
        mission_input, "sha256", "artifact mission-config input"
    )
    actual_sha256 = _file_sha256(mission_config_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "artifact mission config hash does not match its manifest: "
            f"{mission_config_path}"
        )

    resolved = _required(
        artifact_manifest, "resolved_mission_config", "artifact manifest"
    )
    raw_teams = _required(resolved, "teams", "resolved mission config")
    if not isinstance(raw_teams, dict) or not raw_teams:
        raise ValueError("resolved mission config teams must be a non-empty mapping")
    team_grid_configs = {}
    team_member_ids = {}
    grid_orders = {}
    for raw_team_id, raw_team in raw_teams.items():
        team_id = int(raw_team_id)
        team_grid_configs[team_id] = dict(
            _required(raw_team, "grid_config", f"resolved team {team_id}")
        )
        team_member_ids[team_id] = tuple(
            int(value)
            for value in _required(raw_team, "agent_ids", f"resolved team {team_id}")
        )
        grid_orders[team_id] = tuple(
            int(value)
            for value in _required(raw_team, "grid_order", f"resolved team {team_id}")
        )
    coverage = _required(resolved, "coverage", "resolved mission config")
    values = {
        "FRONTIER_TEAM_GRID_CONFIGS": team_grid_configs,
        "FRONTIER_TEAM_MEMBER_IDS": team_member_ids,
        "FRONTIER_GRID_ORDERS_BY_TEAM": grid_orders,
        "FRONTIER_GRID_BOUNDS": {
            int(key): value
            for key, value in _required(
                resolved, "grid_bounds", "resolved mission config"
            ).items()
        },
        "FRONTIER_DYNAMIC_OUTLINE_ENABLED": bool(
            _required(
                resolved,
                "dynamic_outlines_enabled",
                "resolved mission config",
            )
        ),
        "FRONTIER_DYNAMIC_OUTLINE_GRIDS": tuple(
            int(value)
            for value in _required(
                resolved, "dynamic_grid_ids", "resolved mission config"
            )
        ),
        "POST_MISSION_BONXAI_MAP_FILENAME_TEMPLATE": str(
            _required(resolved, "map_filename_template", "resolved mission config")
        ),
        "POST_MISSION_COVERAGE_Z_BOUNDS_M": tuple(
            _required(coverage, "z_bounds_m", "resolved coverage config")
        ),
        "POST_MISSION_COVERAGE_VOXEL_SIZE_M": float(
            _required(coverage, "voxel_size_m", "resolved coverage config")
        ),
        "POST_MISSION_COVERAGE_MATCH_TOLERANCE_M": float(
            _required(coverage, "match_tolerance_m", "resolved coverage config")
        ),
        "POST_MISSION_COVERAGE_VIS": bool(
            _required(coverage, "vis", "resolved coverage config")
        ),
    }
    return type("FrozenMissionConfig", (), values)(), mission_config_path


def _positive_ids(value):
    return _integer_ids(value, "agent", minimum=1)


def _team_ids(value):
    return _integer_ids(value, "team", minimum=0)


def _integer_ids(value, label, minimum):
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{label} IDs must be comma-separated integers"
        ) from exc
    if (
        not result
        or any(item < minimum for item in result)
        or len(result) != len(set(result))
    ):
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise argparse.ArgumentTypeError(
            f"{label} IDs must be unique {qualifier} integers"
        )
    return result


def _load_artifact_manifest(artifact_run):
    supplied_path = Path(artifact_run)
    if supplied_path.is_dir():
        artifact_run = supplied_path.resolve()
    else:
        artifact_run = (DEFAULT_ARTIFACT_ROOT / supplied_path).resolve()
    if not artifact_run.is_dir():
        raise FileNotFoundError(
            "artifact run does not exist as a path or under the default artifact root: "
            f"{artifact_run}"
        )
    manifest_path = artifact_run / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"artifact manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be a YAML mapping")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("artifact manifest is missing a non-empty run_id")
    return artifact_run, manifest_path, manifest


def _team_configuration(v_cfg):
    required = (
        "FRONTIER_TEAM_GRID_CONFIGS",
        "FRONTIER_TEAM_MEMBER_IDS",
        "FRONTIER_GRID_ORDERS_BY_TEAM",
        "FRONTIER_GRID_BOUNDS",
    )
    missing = [name for name in required if not hasattr(v_cfg, name)]
    if missing:
        raise ValueError(
            "mission config is not team-aware; missing: " + ", ".join(missing)
        )

    team_configs = {}
    configured_agents = {}
    for raw_team_id in sorted(v_cfg.FRONTIER_TEAM_MEMBER_IDS):
        team_id = int(raw_team_id)
        member_indices = tuple(
            int(value) for value in v_cfg.FRONTIER_TEAM_MEMBER_IDS[raw_team_id]
        )
        grid_order = tuple(
            int(value) for value in v_cfg.FRONTIER_GRID_ORDERS_BY_TEAM[raw_team_id]
        )
        if not member_indices or not grid_order:
            raise ValueError(f"team {team_id} has empty members or grid sequence")
        artifact_agent_ids = tuple(value + 1 for value in member_indices)
        for agent_id in artifact_agent_ids:
            if agent_id in configured_agents:
                raise ValueError(f"agent {agent_id} belongs to multiple teams")
            configured_agents[agent_id] = team_id
        raw_config = v_cfg.FRONTIER_TEAM_GRID_CONFIGS[raw_team_id]
        team_configs[team_id] = {
            "team_id": team_id,
            "agent_ids": artifact_agent_ids,
            "grid_order": grid_order,
            "grid_config": {
                str(key): value
                for key, value in raw_config.items()
                if key != "agent_ids"
            },
        }
    if not team_configs:
        raise ValueError("mission config contains no frontier teams")
    return team_configs, configured_agents


def _artifact_maps(artifact_run, v_cfg, configured_agents):
    maps = {}
    expected_names = set()
    for agent_id in sorted(configured_agents):
        filename = v_cfg.POST_MISSION_BONXAI_MAP_FILENAME_TEMPLATE.format(
            agent_id=agent_id,
            agent_index=agent_id - 1,
        )
        if Path(filename).name != filename:
            raise ValueError(
                "POST_MISSION_BONXAI_MAP_FILENAME_TEMPLATE must produce a filename"
            )
        if filename in expected_names:
            raise ValueError(f"map filename template produces duplicate name {filename!r}")
        expected_names.add(filename)
        path = artifact_run / filename
        if path.is_file():
            maps[agent_id] = path

    unexpected = sorted(
        path.name
        for path in artifact_run.glob("agent*_map.yaml")
        if path.name not in expected_names
    )
    if unexpected:
        raise ValueError(
            "artifact run contains maps for unconfigured agents: " + ", ".join(unexpected)
        )
    if not maps:
        raise FileNotFoundError(f"artifact run contains no configured agent maps: {artifact_run}")
    return maps


def _select_agents(available_maps, team_configs, configured_agents, agent_ids, team_ids):
    if agent_ids is not None and team_ids is not None:
        raise ValueError("select either agents or teams, not both")
    if agent_ids is not None:
        unknown = sorted(set(agent_ids) - set(configured_agents))
        if unknown:
            raise ValueError(f"unknown configured agent IDs: {unknown}")
        selected = list(agent_ids)
    elif team_ids is not None:
        unknown = sorted(set(team_ids) - set(team_configs))
        if unknown:
            raise ValueError(f"unknown team IDs: {unknown}")
        selected = [
            agent_id
            for team_id in team_ids
            for agent_id in team_configs[team_id]["agent_ids"]
        ]
    else:
        selected = sorted(available_maps)

    missing = [agent_id for agent_id in selected if agent_id not in available_maps]
    if missing:
        raise FileNotFoundError(f"requested artifact maps are missing for agents: {missing}")
    return sorted(selected)


def _bounds3(bounds, z_min, z_max, grid_id):
    try:
        lower = [float(bounds["min"][0]), float(bounds["min"][1]), z_min]
        upper = [float(bounds["max"][0]), float(bounds["max"][1]), z_max]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"grid {grid_id} has malformed XY bounds") from exc
    if any(low >= high for low, high in zip(lower, upper)):
        raise ValueError(f"grid {grid_id} has invalid evaluation bounds")
    return {"min": lower, "max": upper}


def _map_grid_metadata(path):
    """Read Bonxai grid metadata without parsing the potentially huge cell list."""
    values = {}
    wanted = {"resolution", "global_origin", "global_voxels"}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            for key in wanted - values.keys():
                prefix = f"{key}:"
                if stripped.startswith(prefix):
                    try:
                        values[key] = yaml.safe_load(stripped[len(prefix):].strip())
                    except yaml.YAMLError as exc:
                        raise ValueError(f"{path}: malformed Bonxai field {key}") from exc
                    break
            if wanted.issubset(values):
                break
    if not wanted.issubset(values):
        missing = sorted(wanted - values.keys())
        raise ValueError(f"{path}: missing Bonxai grid metadata: {missing}")
    try:
        resolution = float(values["resolution"])
        origin = [float(value) for value in values["global_origin"]]
        voxels = [int(value) for value in values["global_voxels"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed Bonxai grid metadata") from exc
    if resolution <= 0.0 or len(origin) != 3 or len(voxels) != 3:
        raise ValueError(f"{path}: invalid Bonxai grid metadata")
    return resolution, origin, voxels


def _aligned_lower_bound(value, map_origin, resolution):
    steps = math.floor((float(value) - float(map_origin)) / resolution + 1e-9)
    return float(map_origin + steps * resolution)


def snapshot(artifact_run, output_root, ground_truth, mission_config=None,
             agent_ids=None, team_ids=None, run_id=None, copy_maps=False):
    artifact_run, artifact_manifest_path, artifact_manifest = (
        _load_artifact_manifest(artifact_run)
    )
    if mission_config is None:
        v_cfg, mission_config = _frozen_mission_config(
            artifact_run, artifact_manifest
        )
    else:
        if (
            "resolved_mission_config" in artifact_manifest
            or "mission_config" in (artifact_manifest.get("inputs") or {})
        ):
            raise ValueError(
                "--mission-config cannot override a frozen artifact manifest"
            )
        mission_config = Path(mission_config).resolve()
        v_cfg = _load_mission_config(mission_config)
    team_configs, configured_agents = _team_configuration(v_cfg)
    available_maps = _artifact_maps(artifact_run, v_cfg, configured_agents)
    selected_agents = _select_agents(
        available_maps,
        team_configs,
        configured_agents,
        agent_ids,
        team_ids,
    )
    selected_team_ids = sorted({configured_agents[value] for value in selected_agents})

    output_root = Path(output_root).resolve()
    ground_truth = Path(ground_truth).resolve()
    if not ground_truth.is_file():
        raise FileNotFoundError(f"ground truth does not exist: {ground_truth}")

    source_run_id = artifact_manifest["run_id"]
    run_id = run_id or f"{source_run_id}_coverage"
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    agent_maps = []
    if copy_maps:
        maps_dir = run_dir / "maps"
        maps_dir.mkdir()
        for agent_id in selected_agents:
            destination = maps_dir / f"agent{agent_id:03d}_final.yaml"
            shutil.copy2(available_maps[agent_id], destination)
            agent_maps.append(os.path.relpath(destination, run_dir))
    else:
        agent_maps = [
            os.path.relpath(available_maps[agent_id], run_dir)
            for agent_id in selected_agents
        ]
    shutil.copy2(artifact_manifest_path, run_dir / "artifact_manifest.yaml")
    shutil.copy2(mission_config, run_dir / "v_configs.py")

    grid_outcome_paths = {}
    selected_team_agents = sorted({
        agent_id
        for team_id in selected_team_ids
        for agent_id in team_configs[team_id]["agent_ids"]
    })
    for agent_id in selected_team_agents:
        source = artifact_run / f"agent{agent_id:03d}_grid_outcomes.json"
        if not source.is_file():
            continue
        destination_dir = run_dir / "grid_outcomes"
        destination_dir.mkdir(exist_ok=True)
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        grid_outcome_paths[str(agent_id)] = os.path.relpath(destination, run_dir)

    try:
        z_min, z_max = (
            float(value) for value in v_cfg.POST_MISSION_COVERAGE_Z_BOUNDS_M
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "POST_MISSION_COVERAGE_Z_BOUNDS_M must contain two numbers"
        ) from exc
    if z_min >= z_max:
        raise ValueError("post-mission coverage Z bounds are invalid")

    map_resolution, map_origin, _map_voxels = _map_grid_metadata(
        available_maps[selected_agents[0]]
    )
    voxel_size = float(v_cfg.POST_MISSION_COVERAGE_VOXEL_SIZE_M)
    if not math.isclose(map_resolution, voxel_size, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "mission coverage voxel size does not match the Bonxai map resolution: "
            f"{voxel_size} vs {map_resolution}"
        )

    ordered_grid_ids = []
    for team_id in selected_team_ids:
        for grid_id in team_configs[team_id]["grid_order"]:
            if grid_id not in ordered_grid_ids:
                ordered_grid_ids.append(grid_id)

    regions = {}
    for grid_id in ordered_grid_ids:
        if grid_id not in v_cfg.FRONTIER_GRID_BOUNDS:
            raise ValueError(f"grid {grid_id} is missing from FRONTIER_GRID_BOUNDS")
        regions[f"grid_{grid_id}"] = _bounds3(
            v_cfg.FRONTIER_GRID_BOUNDS[grid_id], z_min, z_max, grid_id
        )
    min_x = _aligned_lower_bound(
        min(value["min"][0] for value in regions.values()),
        map_origin[0],
        map_resolution,
    )
    min_y = _aligned_lower_bound(
        min(value["min"][1] for value in regions.values()),
        map_origin[1],
        map_resolution,
    )
    max_x = max(value["max"][0] for value in regions.values())
    max_y = max(value["max"][1] for value in regions.values())

    region_unions = {
        "configured_grid_footprint": [
            f"grid_{grid_id}" for grid_id in ordered_grid_ids
        ]
    }
    manifest_teams = []
    for team_id in selected_team_ids:
        team = team_configs[team_id]
        team_region_names = [f"grid_{value}" for value in team["grid_order"]]
        region_unions[f"team_{team_id}_grid_footprint"] = team_region_names
        manifest_teams.append({
            "team_id": team_id,
            "agent_ids": list(team["agent_ids"]),
            "evaluated_agent_ids": [
                value for value in selected_agents if configured_agents[value] == team_id
            ],
            "grid_config": team["grid_config"],
            "frontier_grid_order": list(team["grid_order"]),
        })

    manifest = {
        "run_id": run_id,
        "artifact_run": {
            "run_id": source_run_id,
            "manifest": "artifact_manifest.yaml",
            "maps_copied": bool(copy_maps),
        },
        "ground_truth": os.path.relpath(ground_truth, run_dir),
        "agent_maps": agent_maps,
        "evaluation": {
            "frame_id": "map",
            "voxel_size_m": float(v_cfg.POST_MISSION_COVERAGE_VOXEL_SIZE_M),
            "match_tolerance_m": float(
                v_cfg.POST_MISSION_COVERAGE_MATCH_TOLERANCE_M
            ),
            "vis": bool(v_cfg.POST_MISSION_COVERAGE_VIS),
            "bounds": {
                "min": [min_x, min_y, z_min],
                "max": [max_x, max_y, z_max],
            },
            "regions": regions,
            "region_unions": region_unions,
            "primary_region": "configured_grid_footprint",
        },
        "ground_truth_transform": {
            "translation_xyz": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "mission_config": {
            "source": "v_configs.py",
            "selected_agent_ids": selected_agents,
            "dynamic_outlines_enabled": bool(
                v_cfg.FRONTIER_DYNAMIC_OUTLINE_ENABLED
            ),
            "dynamic_grid_ids": [
                int(value) for value in v_cfg.FRONTIER_DYNAMIC_OUTLINE_GRIDS
            ],
            "teams": manifest_teams,
        },
        "mission_results": {
            "grid_outcomes": grid_outcome_paths,
        },
    }
    with (run_dir / "manifest.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-run",
        required=True,
        help="run ID or artifact run directory",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--agents", type=_positive_ids, help="one-based IDs, e.g. 1,2,3")
    selection.add_argument("--teams", type=_team_ids, help="zero-based IDs, e.g. 0,1")
    parser.add_argument("--output-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH))
    parser.add_argument(
        "--mission-config",
        help=(
            "explicit config override for a legacy artifact; new artifacts use "
            "their frozen manifest/config and do not consult the working tree"
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--copy-maps",
        action="store_true",
        help="copy maps into the snapshot instead of referencing artifact files",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate immediately and write <snapshot>/results",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    vis_group = parser.add_mutually_exclusive_group()
    vis_group.add_argument("--vis", dest="vis", action="store_true")
    vis_group.add_argument("--no-vis", dest="vis", action="store_false")
    parser.set_defaults(vis=None)
    args = parser.parse_args()
    if not args.evaluate and (
        args.no_plots or args.rebuild_cache or args.vis is not None
    ):
        parser.error("--no-plots, --rebuild-cache, --vis, and --no-vis require --evaluate")
    try:
        run_dir = snapshot(
            args.artifact_run,
            args.output_root,
            args.ground_truth,
            args.mission_config,
            args.agents,
            args.teams,
            args.run_id,
            args.copy_maps,
        )
        print(f"coverage snapshot: {run_dir}")
        if args.evaluate:
            from evaluate_coverage import evaluate
            evaluate(
                run_dir / "manifest.yaml",
                run_dir / "results",
                plots=not args.no_plots,
                rebuild_cache=args.rebuild_cache,
                vis_override=args.vis,
            )
            print(f"coverage results: {run_dir / 'results'}")
    except Exception as exc:
        print(f"coverage snapshot failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
