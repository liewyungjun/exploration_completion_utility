import hashlib
import json
import numpy as np
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


UTILITY_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(UTILITY_SRC))

from snapshot_configured_run import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_GROUND_TRUTH,
    snapshot,
)


CONFIG_SOURCE = """
FRONTIER_TEAM_GRID_CONFIGS = {
    0: {"agent_ids": [0, 1], "source": "simple_sector_sequence", "sequence": [10, 11]},
    1: {"agent_ids": [2], "source": "simple_sector_road", "road": "middle"},
}
FRONTIER_TEAM_MEMBER_IDS = {0: (0, 1), 1: (2,)}
FRONTIER_GRID_ORDERS_BY_TEAM = {0: (10, 11), 1: (20, 11)}
FRONTIER_GRID_BOUNDS = {
    10: {"min": [0.0, 0.0, 0.0], "max": [2.0, 2.0, 0.0]},
    11: {"min": [2.0, 0.0, 0.0], "max": [4.0, 2.0, 0.0]},
    20: {"min": [0.0, 2.0, 0.0], "max": [4.0, 4.0, 0.0]},
}
FRONTIER_DYNAMIC_OUTLINE_ENABLED = True
FRONTIER_DYNAMIC_OUTLINE_GRIDS = [11, 20]
POST_MISSION_BONXAI_MAP_FILENAME_TEMPLATE = "agent{agent_id:03d}_map.yaml"
POST_MISSION_COVERAGE_Z_BOUNDS_M = (0.0, 6.0)
POST_MISSION_COVERAGE_VOXEL_SIZE_M = 0.2
POST_MISSION_COVERAGE_MATCH_TOLERANCE_M = 0.3
POST_MISSION_COVERAGE_VIS = False
"""


class SnapshotConfiguredRunTests(unittest.TestCase):
    def test_defaults_follow_the_sibling_repository_layout(self):
        workspace_src = Path(__file__).resolve().parents[2]
        self.assertEqual(
            DEFAULT_ARTIFACT_ROOT,
            workspace_src / "flush_search" / "artifacts",
        )
        self.assertEqual(
            DEFAULT_GROUND_TRUTH,
            workspace_src
            / "exploration_completion_utility"
            / "resources"
            / "virtualrun_2.ply",
        )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.artifact_run = self.root / "artifacts" / "run_a"
        self.artifact_run.mkdir(parents=True)
        (self.artifact_run / "manifest.yaml").write_text(
            "run_id: run_a\nstarted_utc: 2026-08-04T00:00:00Z\n",
            encoding="utf-8",
        )
        self.config = self.root / "v_configs.py"
        self.config.write_text(CONFIG_SOURCE, encoding="utf-8")
        self.ground_truth = self.root / "ground_truth.ply"
        self.ground_truth.write_text("ply\n", encoding="utf-8")
        self.output_root = self.root / "coverage"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _map(self, agent_id):
        path = self.artifact_run / f"agent{agent_id:03d}_map.yaml"
        path.write_text(
            "bonxai_probabilistic_map:\n"
            "  resolution: 0.2\n"
            "  global_origin: [-50.0, -50.0, 0.0]\n"
            "  global_voxels: [500, 500, 30]\n"
            f"  agent: {agent_id}\n",
            encoding="utf-8",
        )
        return path

    def _freeze_mission_config(self):
        frozen_config = self.artifact_run / "v_configs.py"
        frozen_config.write_text(CONFIG_SOURCE, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "run_id": "run_a",
            "inputs": {
                "mission_config": {
                    "path": "v_configs.py",
                    "sha256": hashlib.sha256(frozen_config.read_bytes()).hexdigest(),
                }
            },
            "resolved_mission_config": {
                "teams": {
                    0: {
                        "agent_ids": [0, 1],
                        "grid_order": [10, 11],
                        "grid_config": {
                            "agent_ids": [0, 1],
                            "source": "simple_sector_sequence",
                            "sequence": [10, 11],
                        },
                    },
                    1: {
                        "agent_ids": [2],
                        "grid_order": [20, 11],
                        "grid_config": {
                            "agent_ids": [2],
                            "source": "simple_sector_road",
                            "road": "middle",
                        },
                    },
                },
                "grid_bounds": {
                    10: {"min": [0.0, 0.0, 0.0], "max": [2.0, 2.0, 0.0]},
                    11: {"min": [2.0, 0.0, 0.0], "max": [4.0, 2.0, 0.0]},
                    20: {"min": [0.0, 2.0, 0.0], "max": [4.0, 4.0, 0.0]},
                },
                "dynamic_outlines_enabled": True,
                "dynamic_grid_ids": [11, 20],
                "map_filename_template": "agent{agent_id:03d}_map.yaml",
                "coverage": {
                    "z_bounds_m": [0.0, 6.0],
                    "voxel_size_m": 0.2,
                    "match_tolerance_m": 0.3,
                    "vis": False,
                },
            },
        }
        (self.artifact_run / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        return frozen_config

    def test_uses_frozen_artifact_config_without_working_tree_fallback(self):
        frozen_config = self._freeze_mission_config()
        self._map(1)
        self._map(3)

        run_dir = snapshot(
            self.artifact_run,
            self.output_root,
            self.ground_truth,
            None,
        )

        self.assertEqual(
            (run_dir / "v_configs.py").read_bytes(), frozen_config.read_bytes()
        )
        with (run_dir / "manifest.yaml").open(encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream)
        self.assertEqual(
            manifest["evaluation"]["region_unions"]["configured_grid_footprint"],
            ["grid_10", "grid_11", "grid_20"],
        )

    def test_missing_frozen_config_requires_explicit_legacy_override(self):
        with self.assertRaisesRegex(ValueError, "inputs"):
            snapshot(
                self.artifact_run,
                self.output_root,
                self.ground_truth,
                None,
            )

    def test_explicit_config_cannot_override_frozen_artifact(self):
        self._freeze_mission_config()
        with self.assertRaisesRegex(ValueError, "cannot override"):
            snapshot(
                self.artifact_run,
                self.output_root,
                self.ground_truth,
                self.config,
            )

    def test_discovers_partial_multi_team_maps_and_references_them(self):
        map_1 = self._map(1)
        map_3 = self._map(3)
        outcome = self.artifact_run / "agent001_grid_outcomes.json"
        outcome.write_text(
            '{"grid_outcomes":{"10":{"status":"local_complete"}}}\n',
            encoding="utf-8",
        )

        run_dir = snapshot(
            self.artifact_run,
            self.output_root,
            self.ground_truth,
            self.config,
        )

        with (run_dir / "manifest.yaml").open(encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream)
        resolved_maps = [
            (run_dir / value).resolve() for value in manifest["agent_maps"]
        ]
        self.assertEqual(resolved_maps, [map_1.resolve(), map_3.resolve()])
        self.assertFalse(manifest["artifact_run"]["maps_copied"])
        self.assertEqual(
            manifest["evaluation"]["region_unions"]["configured_grid_footprint"],
            ["grid_10", "grid_11", "grid_20"],
        )
        self.assertEqual(
            manifest["evaluation"]["region_unions"]["team_1_grid_footprint"],
            ["grid_20", "grid_11"],
        )
        self.assertEqual(
            [team["evaluated_agent_ids"] for team in manifest["mission_config"]["teams"]],
            [[1], [3]],
        )
        self.assertTrue((run_dir / "artifact_manifest.yaml").is_file())
        self.assertTrue((run_dir / "v_configs.py").is_file())
        self.assertEqual(
            manifest["mission_config"]["dynamic_grid_ids"], [11, 20]
        )
        self.assertEqual(
            manifest["mission_results"]["grid_outcomes"],
            {"1": "grid_outcomes/agent001_grid_outcomes.json"},
        )
        self.assertEqual(
            (run_dir / "grid_outcomes" / outcome.name).read_bytes(),
            outcome.read_bytes(),
        )

    def test_team_selection_requires_every_team_map(self):
        self._map(1)
        with self.assertRaisesRegex(FileNotFoundError, r"agents: \[2\]"):
            snapshot(
                self.artifact_run,
                self.output_root,
                self.ground_truth,
                self.config,
                team_ids=[0],
            )

    def test_copy_maps_makes_self_contained_snapshot(self):
        self._map(3)
        run_dir = snapshot(
            self.artifact_run,
            self.output_root,
            self.ground_truth,
            self.config,
            agent_ids=[3],
            copy_maps=True,
        )
        with (run_dir / "manifest.yaml").open(encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream)
        self.assertEqual(manifest["agent_maps"], ["maps/agent003_final.yaml"])
        self.assertTrue(manifest["artifact_run"]["maps_copied"])
        self.assertTrue((run_dir / "maps" / "agent003_final.yaml").is_file())

    def test_snapshot_adds_only_selected_second_floor_regions(self):
        self.config.write_text(CONFIG_SOURCE +
            "\nFRONTIER_SECOND_FLOOR_DYNAMIC_GRIDS = [20, 21]\n"
            "FRONTIER_FIRST_FLOOR_VOLUME_Z_RANGE_M = (1, 3)\n"
            "FRONTIER_SECOND_FLOOR_VOLUME_Z_RANGE_M = (5, 6)\n")
        self._map(3)
        run_dir = snapshot(self.artifact_run, self.output_root, self.ground_truth,
                           self.config, agent_ids=[3])
        manifest = yaml.safe_load((run_dir / 'manifest.yaml').read_text())
        evaluation = manifest['evaluation']
        self.assertEqual(evaluation['additional_plot_regions'], ['second_floor_grid_footprint'])
        self.assertEqual(evaluation['region_unions']['second_floor_grid_footprint'],
                         ['grid_20_second_floor'])
        self.assertEqual(evaluation['regions']['grid_20_second_floor']['min'][2], 3)
        self.assertEqual(evaluation['regions']['grid_20_second_floor']['max'][2], 6)
        self.assertNotIn('grid_21_second_floor', evaluation['regions'])
        self.assertEqual(evaluation['bounds']['max'][2], 6)

    def test_npz_snapshot_reads_header_and_preserves_extension(self):
        self.config.write_text(CONFIG_SOURCE.replace('_map.yaml', '_map.npz'))
        path = self.artifact_run / 'agent003_map.npz'
        np.savez(path, schema_version=np.int32(2), metadata_json=json.dumps(dict(
            resolution=0.2, global_origin=[-50., -50., 0.], global_voxels=[500, 500, 30])))
        run_dir = snapshot(self.artifact_run, self.output_root, self.ground_truth,
                           self.config, agent_ids=[3], copy_maps=True)
        manifest = yaml.safe_load((run_dir / 'manifest.yaml').read_text())
        self.assertEqual(manifest['agent_maps'], ['maps/agent003_final.npz'])
        self.assertEqual((run_dir / 'maps/agent003_final.npz').read_bytes(), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
