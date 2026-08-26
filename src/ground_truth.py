"""Streaming ground-truth PLY transformation, cropping, and voxelisation."""

from pathlib import Path

import numpy as np


_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _header(path):
    properties = []
    vertex_count = None
    vertex_element = False
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PLY header has no end_header")
            line = raw.decode("ascii").strip()
            fields = line.split()
            if fields[:1] == ["format"]:
                fmt = fields[1]
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                vertex_element = True
            elif fields[:1] == ["element"]:
                vertex_element = False
            elif fields[:1] == ["property"] and vertex_element:
                if fields[1] == "list":
                    raise ValueError("list property is not valid in a PLY vertex element")
                properties.append((fields[2], fields[1]))
            elif line == "end_header":
                return fmt, vertex_count, properties, stream.tell()


def _rotation(quaternion_xyzw):
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0:
        raise ValueError("ground-truth quaternion must be non-zero")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.asarray([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def load_voxels(
    path, origin, voxel_size, bounds_min, bounds_max, translation, quaternion,
    chunk_size=1_000_000,
):
    """Stream PLY vertices and return cropped, transformed reference voxel keys."""
    fmt, count, properties, offset = _header(path)
    if count is None or not all(name in dict(properties) for name in ("x", "y", "z")):
        raise ValueError("PLY must contain a vertex element with x, y, z properties")
    endian = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(fmt)
    if endian is None:
        raise ValueError("only binary little/big-endian PLY files are supported")
    try:
        dtype = np.dtype([(name, endian + _PLY_TYPES[kind]) for name, kind in properties])
    except KeyError as exc:
        raise ValueError(f"unsupported PLY property type: {exc.args[0]}") from exc
    vertices = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(count,))
    rotation = _rotation(quaternion)
    translation = np.asarray(translation, dtype=np.float64)
    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    if np.any(lower >= upper):
        raise ValueError("evaluation bounds min must be less than max")
    keys = set()
    origin = np.asarray(origin, dtype=np.float64)
    for start in range(0, count, chunk_size):
        block = vertices[start:min(start + chunk_size, count)]
        points = np.column_stack((block["x"], block["y"], block["z"])).astype(
            np.float64, copy=False
        )
        points = points @ rotation.T + translation
        points = points[np.all(points >= lower, axis=1) & np.all(points < upper, axis=1)]
        if len(points):
            quantised = np.floor((points - origin) / voxel_size).astype(np.int64)
            keys.update(map(tuple, np.unique(quantised, axis=0)))
    if not keys:
        raise ValueError("cropped ground-truth point cloud is empty")
    return keys
