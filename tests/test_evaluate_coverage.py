import sys
import unittest
from pathlib import Path


UTILITY_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(UTILITY_SRC))

from evaluate_coverage import _agent_id_from_map_path


class EvaluateCoverageTests(unittest.TestCase):
    def test_agent_id_accepts_artifact_and_snapshot_names(self):
        self.assertEqual(_agent_id_from_map_path(Path("agent001_map.yaml")), "001")
        self.assertEqual(_agent_id_from_map_path(Path("agent001_final.yaml")), "001")
        self.assertEqual(_agent_id_from_map_path(Path("agents002_map.yaml")), "002")

    def test_unrecognised_map_name_is_preserved(self):
        self.assertEqual(_agent_id_from_map_path(Path("leader.yaml")), "leader")


if __name__ == "__main__":
    unittest.main()
