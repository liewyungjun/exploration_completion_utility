import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from snapshot_configured_run import _second_floor_settings
from evaluate_coverage import evaluate


class SecondFloorCoverageTests(unittest.TestCase):
    def test_only_selected_configured_second_floor_grids_are_included(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'v_configs.py'
            path.write_text("raise RuntimeError('must not execute')\n"
                'FRONTIER_SECOND_FLOOR_DYNAMIC_GRIDS = [20, 21]\n'
                'FRONTIER_FIRST_FLOOR_VOLUME_Z_RANGE_M = (1, 3)\n'
                'FRONTIER_SECOND_FLOOR_VOLUME_Z_RANGE_M = (5, 6)\n')
            self.assertEqual(_second_floor_settings(path, [1, 21])['grid_ids'], [21])
            self.assertEqual(_second_floor_settings(path, [21])['coverage_z_bounds_m'], [3, 6])
            self.assertIsNone(_second_floor_settings(path, [1]))

    def test_second_floor_outside_primary_height_is_loaded_and_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            gt = root / 'gt.ply'
            header = ('ply\nformat binary_little_endian 1.0\nelement vertex 2\n'
                      'property float x\nproperty float y\nproperty float z\nend_header\n')
            gt.write_bytes(header.encode() + struct.pack('<ffffff', .1, .1, .1, .1, .1, 3.1))
            indices = np.array([[0, 0, 0], [0, 0, 15]], dtype=np.int32)
            metadata = {'frame_id': 'map', 'resolution': .2, 'global_origin': [0, 0, 0],
                'global_voxels': [5, 5, 30], 'counts': {'active_cells': 2,
                'occupied_cells': 2, 'free_cells': 0, 'active_unknown_cells': 0},
                'bonxai_options': {'occupancy_threshold_log': 0}}
            np.savez(root / 'agent001_map.npz', schema_version=np.int32(2),
                metadata_json=json.dumps(metadata), indices=indices,
                positions=(indices.astype(float) + .5) * .2,
                probability_log=np.array([1, 1], dtype=np.int32))
            lower={'min':[0,0,0], 'max':[1,1,3]}
            upper={'min':[0,0,3], 'max':[1,1,6]}
            manifest={'run_id':'test', 'ground_truth':'gt.ply', 'agent_maps':['agent001_map.npz'],
                'ground_truth_transform': {'translation_xyz':[0,0,0], 'quaternion_xyzw':[0,0,0,1]},
                'evaluation': {'frame_id':'map', 'voxel_size_m':.2, 'match_tolerance_m':.1,
                    'bounds':lower, 'regions':{'grid_20':lower, 'grid_20_second_floor':upper},
                    'region_unions':{'second_floor_grid_footprint':['grid_20_second_floor']},
                    'additional_plot_regions':['second_floor_grid_footprint']}}
            path=root/'manifest.yaml';path.write_text(yaml.safe_dump(manifest))
            result=evaluate(path, root/'results', vis_override=False)
            self.assertEqual(result['regions']['whole_environment']['ground_truth_voxels'], 1)
            self.assertEqual(result['regions']['second_floor_grid_footprint']['matched_ground_truth_voxels'], 1)
            self.assertEqual(result['regions']['second_floor_grid_footprint']['recall'], 1)
            self.assertTrue((root/'results/plots/coverage_second_floor_top_down.png').is_file())
            self.assertIn('## Second-floor coverage', (root/'results/coverage_summary.md').read_text())
            self.assertIn('grid_20_second_floor', (root/'results/region_metrics.csv').read_text())


if __name__ == '__main__':
    unittest.main()
