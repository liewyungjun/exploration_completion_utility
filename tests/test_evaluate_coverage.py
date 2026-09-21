import sys
import unittest
from pathlib import Path


UTILITY_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(UTILITY_SRC))

from evaluate_coverage import _agent_id_from_map_path, _literal_config_value


class EvaluateCoverageTests(unittest.TestCase):
    def test_agent_id_accepts_artifact_and_snapshot_names(self):
        self.assertEqual(_agent_id_from_map_path(Path("agent001_map.yaml")), "001")
        self.assertEqual(_agent_id_from_map_path(Path("agent001_final.yaml")), "001")
        self.assertEqual(_agent_id_from_map_path(Path("agents002_map.yaml")), "002")

    def test_unrecognised_map_name_is_preserved(self):
        self.assertEqual(_agent_id_from_map_path(Path("leader.yaml")), "leader")

    def test_reads_legacy_dynamic_grid_literal_without_importing_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v_configs.py"
            path.write_text(
                "raise RuntimeError('must not execute')\n"
                "FRONTIER_DYNAMIC_OUTLINE_GRIDS = [20, 21, 22, 23]\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _literal_config_value(path, "FRONTIER_DYNAMIC_OUTLINE_GRIDS"),
                [20, 21, 22, 23],
            )


if __name__ == "__main__":
    unittest.main()
