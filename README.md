# Exploration Completion Utility

Offline evaluation for saved Bonxai occupancy maps against a reference point cloud.

## Directory layout

```text
exploration_completion_utility/
├── resources/
│   ├── virtualrun.ply                 # default ground-truth cloud
│   ├── manual_maps/                   # maps saved by a manual FSM launch
│   └── manual_runs/                   # timestamped runs from save_final_bonxai_maps.py
├── runs/                              # frozen snapshots of aeromaze-devctl artifacts
├── src/                               # evaluator and snapshot scripts
└── tests/                             # unit tests
```

`resources/` contains reusable inputs and manually generated data; `runs/`
contains the evaluator's reproducible analysis snapshots.

## Artifact run: snapshot and evaluate

Run these commands from the workspace `src/` directory:

```bash
PYTHONPATH=detect_stairs/src \
python3 exploration_completion_utility/src/snapshot_configured_run.py \
  --artifact-run <artifact_run_id> \
  --evaluate \
  --no-vis
```

`<artifact_run_id>` is resolved under
`pybullet_e2e/ros_pybullet_gym/artifacts/` (or provide its full path). The
new snapshot and its results are written to
`exploration_completion_utility/runs/<artifact_run_id>_coverage/` by default.
Use `--teams 0,1` or `--agents 1,4,6` to select maps, and `--run-id` to choose
the snapshot directory name.

To snapshot without evaluating, omit `--evaluate`. Evaluate it later with:

```bash
PYTHONPATH=detect_stairs/src \
python3 exploration_completion_utility/src/evaluate_coverage.py \
  --manifest exploration_completion_utility/runs/<snapshot_id>/manifest.yaml \
  --output exploration_completion_utility/runs/<snapshot_id>/results \
  --no-vis
```

The default reference is `resources/virtualrun.ply`. Artifact maps are
referenced in place by default; pass `--copy-maps` only when a snapshot needs
its own map copies.

### Re-run an existing snapshot with interactive visualisation

If the artifact run has already been snapshotted, evaluated, and cached, do
not snapshot it again. Re-run the evaluator against the existing snapshot and
add `--vis`:

```bash
PYTHONPATH=detect_stairs/src \
python3 exploration_completion_utility/src/evaluate_coverage.py \
  --manifest exploration_completion_utility/runs/<snapshot_id>/manifest.yaml \
  --output exploration_completion_utility/runs/<snapshot_id>/results \
  --vis
```

Replace `<snapshot_id>` with the existing snapshot directory name, for
example `20260806T134652_fsm_coverage`. The existing map cache is reused by
default; add `--rebuild-cache` only if the referenced map files have changed.
Close the 3D viewer window to let the command finish.

## Manual runs (without `aeromaze-devctl`)

The waypoint controller uses `RUN_ARTIFACT_DIR` when it is set by
`aeromaze-devctl`. When it is unset during a manual launch, final per-agent
maps now go to:

```text
exploration_completion_utility/resources/manual_maps/agentNNN_map.yaml
```

This location is intentionally persistent but filenames are per agent, so a
later manual launch replaces that agent's previous map. For a timestamped,
self-contained manual collection, use `save_final_bonxai_maps.py`; its default
destination is `resources/manual_runs/<timestamp>_run/` and it records a
manifest alongside its `maps/` directory.

## Results

Each evaluated snapshot writes `results/` with `coverage_summary.md`,
`coverage_summary.json`, `agent_metrics.csv`, `region_metrics.csv`, plots, and
reusable map caches.

## Tests

```bash
python3 -m unittest discover \
  -s exploration_completion_utility/tests \
  -p 'test_*.py' -v
```

See [README_EXPLANATION.md](README_EXPLANATION.md) for the data contract and
metric definitions.
