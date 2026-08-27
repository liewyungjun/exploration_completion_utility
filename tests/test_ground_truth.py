#!/usr/bin/env python3

import struct
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(MODULE_DIR))

from ground_truth import load_voxels


class GroundTruthTest(unittest.TestCase):
    def test_binary_ply_transform_crop_and_voxelise(self):
        header = (
            b"ply\nformat binary_little_endian 1.0\n"
            b"element vertex 3\nproperty float x\nproperty float y\n"
            b"property float z\nend_header\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ply"
            with path.open("wb") as stream:
                stream.write(header)
                for point in ((0.1, 0.1, 0.1), (0.19, 0.1, 0.1), (2.0, 2.0, 2.0)):
                    stream.write(struct.pack("<fff", *point))
            keys = load_voxels(
                path, [1, 0, 0], 0.2, [1, 0, 0], [2, 1, 1],
                [1, 0, 0], [0, 0, 0, 1],
            )
        self.assertEqual(keys, {(0, 0, 0)})

    def test_empty_crop_fails(self):
        header = (
            b"ply\nformat binary_little_endian 1.0\n"
            b"element vertex 1\nproperty float x\nproperty float y\n"
            b"property float z\nend_header\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ply"
            path.write_bytes(header + struct.pack("<fff", 10, 10, 10))
            with self.assertRaisesRegex(ValueError, "empty"):
                load_voxels(
                    path, [0, 0, 0], 0.2, [0, 0, 0], [1, 1, 1],
                    [0, 0, 0], [0, 0, 0, 1],
                )


if __name__ == "__main__":
    unittest.main()
