#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(MODULE_DIR))

from evaluate_coverage import _visualisation_enabled
from report import _plot_grid_regions, _voxel_centres, _voxel_marker_area


class VisualisationTest(unittest.TestCase):
    def test_manifest_value_and_cli_override(self):
        self.assertTrue(_visualisation_enabled({"vis": True}, None))
        self.assertFalse(_visualisation_enabled({"vis": True}, False))
        self.assertTrue(_visualisation_enabled({"vis": False}, True))

    def test_invalid_manifest_value_fails(self):
        with self.assertRaisesRegex(ValueError, "true or false"):
            _visualisation_enabled({"vis": "true"}, None)

    def test_voxel_centres_use_shared_origin(self):
        centres = _voxel_centres({(0, 0, 0), (1, 2, 3)}, [10, 20, 30], 0.2)
        self.assertEqual(centres.shape, (2, 3))
        actual = {tuple(row) for row in np.round(centres, decimals=10)}
        self.assertEqual(actual, {(10.1, 20.1, 30.1), (10.3, 20.5, 30.7)})

    def test_projected_marker_area_tracks_voxel_size(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        lower = [-10, -10, 0]
        upper = [10, 10, 6]
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_zlim(lower[2], upper[2])
        axis.set_box_aspect(np.subtract(upper, lower))
        small = _voxel_marker_area(axis, figure, 0.2, lower, upper)
        large = _voxel_marker_area(axis, figure, 0.4, lower, upper)
        plt.close(figure)

        self.assertGreater(small, 0)
        self.assertGreater(large, small)
        self.assertAlmostEqual(large / small, 4.0, delta=0.2)

    def test_grid_regions_draw_outlines_and_labels(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        _plot_grid_regions(
            axis,
            {"grid_7": [([0.0, 1.0, 0.0], [2.0, 4.0, 1.0])]},
        )

        self.assertEqual(len(axis.lines), 1)
        self.assertEqual([text.get_text() for text in axis.texts], ["Grid 7"])
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
