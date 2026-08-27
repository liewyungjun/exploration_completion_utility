import sys
import tempfile
import unittest
from pathlib import Path

import yaml


UTILITY_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(UTILITY_SRC))

from snapshot_configured_run import DEFAULT_ARTIFACT_ROOT, DEFAULT_CONFIG, snapshot


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
            DEFAULT_CONFIG,
            workspace_src / "flush_search" / "config" / "v_configs.py",
        )
        self.assertEqual(
            DEFAULT_ARTIFACT_ROOT,
            workspace_src / "flush_search" / "artifacts",
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

    def test_discovers_partial_multi_team_maps_and_references_them(self):
        map_1 = self._map(1)
        map_3 = self._map(3)

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


if __name__ == "__main__":
    unittest.main()
