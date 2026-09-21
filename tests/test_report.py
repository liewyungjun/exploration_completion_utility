import sys
import tempfile
import unittest
from pathlib import Path


UTILITY_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(UTILITY_SRC))

from report import write_reports


def _metrics(recall, precision):
    f1 = 2.0 * recall * precision / (recall + precision)
    return {
        "ground_truth_voxels": 100,
        "occupied_voxels": 80,
        "matched_ground_truth_voxels": int(100 * recall),
        "matched_occupied_voxels": int(80 * precision),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


class ReportTests(unittest.TestCase):
    def test_markdown_contains_grid_and_team_agent_tables(self):
        primary = {
            **_metrics(0.8, 0.9),
            "known_voxels": 120,
            "classification_conflicts": 0,
            "classification_conflict_fraction": 0.0,
            "best_agent_recall": 0.7,
            "improvement_over_best_agent": 0.1,
        }
        summary = {
            "run_id": "test_run",
            "manifest": "/tmp/manifest.yaml",
            "agents_evaluated": 1,
            "primary_region": "grid_11",
            "evaluation": {
                "voxel_size_m": 0.2,
                "match_tolerance_m": 0.3,
                "frame_id": "map",
            },
            "regions": {"grid_11": primary},
            "mission": {
                "dynamic_grid_ids": [11],
                "teams": [{
                    "team_id": 0,
                    "agent_ids": [1, 2],
                    "frontier_grid_order": [11],
                }],
                "grid_outcomes": {
                    "1": {"11": {"status": "local_complete"}},
                    "2": {"11": {"status": "incomplete"}},
                },
                "team_grid_metrics": {"0": {"grid_11": _metrics(0.8, 0.9)}},
            },
        }
        agent = {
            **_metrics(0.7, 0.85),
            "agent_id": "001",
            "map": "/tmp/agent001_map.yaml",
            "known_voxels": 100,
            "free_voxels": 20,
            "unique_contribution_voxels": 70,
            "overlap_occupied_voxels": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            write_reports(directory, summary, {"grid_11": [agent]})
            report = (Path(directory) / "coverage_summary.md").read_text(
                encoding="utf-8"
            )

        self.assertIn("## Per-grid statistics", report)
        self.assertIn("### Team 0", report)
        self.assertIn("| 11 | dynamic |", report)
        self.assertIn("1 local_complete", report)
        self.assertIn("1 incomplete", report)
        self.assertIn("## Per-agent grid statistics", report)
        self.assertIn("| Grid | Type | Agent 1 | Agent 2 | Total |", report)
        self.assertIn("—<br><span style=", report)
        self.assertIn("><strong>70.00%</strong></span>", report)


if __name__ == "__main__":
    unittest.main()
