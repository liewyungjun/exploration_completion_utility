#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(MODULE_DIR))

from coverage import (
    aligned_index_keys, evaluate_region, neighbour_offsets, occupied_voxel_metrics,
)


class CoverageTest(unittest.TestCase):
    def test_aligned_integer_indices_preserve_adjacent_float32_voxels(self):
        positions = np.asarray(
            [[-21.8, -19.8, 0.2], [-21.6, -19.8, 0.2]], dtype=np.float32
        )
        float_keys = np.floor(
            (positions.astype(np.float64) - [-22.0, -20.0, 0.0]) / 0.2
        ).astype(np.int64)
        self.assertEqual(len(np.unique(float_keys, axis=0)), 1)

        keys = aligned_index_keys(
            np.asarray([[141, 151, 1], [142, 151, 1]], dtype=np.int32),
            [-50.0, -50.0, 0.0],
            0.2,
            [-22.0, -20.0, 0.0],
            0.2,
        )
        self.assertEqual(keys, {(1, 1, 1), (2, 1, 1)})

    def test_aligned_integer_indices_reject_incompatible_grids(self):
        indices = np.asarray([[0, 0, 0]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "resolution"):
            aligned_index_keys(indices, [0, 0, 0], 0.1, [0, 0, 0], 0.2)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            aligned_index_keys(indices, [0.1, 0, 0], 0.2, [0, 0, 0], 0.2)

    def test_aligned_integer_indices_reject_duplicates(self):
        indices = np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "duplicate map_index"):
            aligned_index_keys(indices, [0, 0, 0], 0.2, [0, 0, 0], 0.2)

    def test_perfect_and_empty_reconstruction(self):
        gt = {(0, 0, 0), (1, 0, 0)}
        offsets = neighbour_offsets(0.2, 0.0)
        self.assertEqual(occupied_voxel_metrics(gt, gt, offsets)["f1"], 1.0)
        self.assertEqual(occupied_voxel_metrics(gt, set(), offsets)["recall"], 0.0)

    def test_complementary_agents_reach_full_recall(self):
        gt = {(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)}
        agents = [
            {"agent_id": "001", "map": "one.yaml",
             "occupied": {(0, 0, 0), (1, 0, 0)}, "free": set()},
            {"agent_id": "002", "map": "two.yaml",
             "occupied": {(2, 0, 0), (3, 0, 0)}, "free": set()},
        ]
        swarm, per_agent = evaluate_region(gt, agents, neighbour_offsets(0.2, 0.0))
        self.assertEqual(swarm["recall"], 1.0)
        self.assertEqual([row["recall"] for row in per_agent], [0.5, 0.5])
        self.assertEqual([row["unique_contribution_voxels"] for row in per_agent], [2, 2])

    def test_false_occupied_voxel_reduces_precision(self):
        metrics = occupied_voxel_metrics(
            {(0, 0, 0)}, {(0, 0, 0), (9, 0, 0)}, neighbour_offsets(0.2, 0.0)
        )
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["precision"], 0.5)

    def test_euclidean_tolerance_boundary(self):
        offsets = neighbour_offsets(0.2, 0.3)
        self.assertIn((1, 1, 0), offsets)
        self.assertNotIn((1, 1, 1), offsets)

    def test_classification_conflict(self):
        agents = [
            {"agent_id": "001", "map": "one", "occupied": {(0, 0, 0)}, "free": set()},
            {"agent_id": "002", "map": "two", "occupied": set(), "free": {(0, 0, 0)}},
        ]
        swarm, _ = evaluate_region(
            {(0, 0, 0)}, agents, neighbour_offsets(0.2, 0.0)
        )
        self.assertEqual(swarm["classification_conflicts"], 1)


if __name__ == "__main__":
    unittest.main()
