from __future__ import annotations

import argparse
import struct
from pathlib import Path

import glfw
import numpy as np
import torch

try:
    from .torch_gl_viewer import TorchGLViewer
except ImportError:
    from torch_gl_viewer import TorchGLViewer


def _parse_ply_header(f):
    line = f.readline()
    if line != b"ply\n":
        raise ValueError("Expected a PLY file")

    vertex_count = None
    strip_count = 0
    vertex_properties = []
    current_element = None

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected end of PLY header")
        text = line.decode("ascii").strip()
        if text == "end_header":
            break
        if text == "format binary_little_endian 1.0":
            continue
        if text.startswith("format "):
            raise ValueError(f"Unsupported PLY format: {text}")
        if text.startswith("element "):
            _, name, count = text.split()
            current_element = name
            if name == "vertex":
                vertex_count = int(count)
            elif name == "tristrips":
                strip_count = int(count)
            continue
        if text.startswith("property ") and current_element == "vertex":
            parts = text.split()
            if len(parts) == 3:
                vertex_properties.append((parts[1], parts[2]))

    if vertex_count is None:
        raise ValueError("PLY file does not declare vertices")
    return vertex_count, strip_count, vertex_properties


def _vertex_dtype(properties):
    dtype_map = {
        "char": "i1",
        "uchar": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "float": "<f4",
        "double": "<f8",
    }
    fields = []
    for scalar_type, name in properties:
        if scalar_type not in dtype_map:
            raise ValueError(f"Unsupported vertex property type: {scalar_type}")
        fields.append((name, dtype_map[scalar_type]))
    return np.dtype(fields)


def _tristrip_to_faces(indices):
    faces = []
    strip = []

    def flush_strip():
        for i in range(len(strip) - 2):
            tri = (strip[i], strip[i + 1], strip[i + 2])
            if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                continue
            if i % 2:
                faces.append((tri[1], tri[0], tri[2]))
            else:
                faces.append(tri)

    for idx in indices:
        if idx < 0:
            flush_strip()
            strip = []
        else:
            strip.append(int(idx))
    flush_strip()
    return np.asarray(faces, dtype=np.int32)


def load_binary_ply_tristrip(path):
    with Path(path).open("rb") as f:
        vertex_count, strip_count, properties = _parse_ply_header(f)
        dtype = _vertex_dtype(properties)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
        vertices = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)

        strips = []
        for _ in range(strip_count):
            raw_count = f.read(4)
            if len(raw_count) != 4:
                raise ValueError("Unexpected end of PLY tristrip data")
            (count,) = struct.unpack("<i", raw_count)
            strip = np.fromfile(f, dtype="<i4", count=count)
            if strip.shape[0] != count:
                raise ValueError("Unexpected end of PLY tristrip indices")
            strips.append(strip)

    if not strips:
        raise ValueError("PLY file does not contain triangle strips")

    faces = _tristrip_to_faces(np.concatenate(strips))
    if faces.size == 0:
        raise ValueError("PLY triangle strips did not produce triangles")
    return vertices, faces


def normalize_vertices(vertices):
    vertices = vertices.astype(np.float32, copy=True)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices -= center
    scale = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    if scale > 0.0:
        vertices /= scale
    return vertices


def main():
    parser = argparse.ArgumentParser(description="Stanford bunny mesh + point cloud demo.")
    parser.add_argument("--ply", type=Path, default=Path(__file__).with_name("bunny.ply"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=2.3)
    parser.add_argument("--vsync", action="store_true")
    args = parser.parse_args()

    vertices_np, faces_np = load_binary_ply_tristrip(args.ply)
    vertices_np = normalize_vertices(vertices_np)

    vis = TorchGLViewer(
        "Stanford Bunny - Mesh and Point Cloud",
        width=args.width,
        height=args.height,
        background=(0.97, 0.97, 0.95),
        vsync=args.vsync,
        cuda_device=args.cuda_device,
        show_fps=True,
        show_controls=True,
    )
    vis.set_lighting(
        direction=(0.35, -0.35, -1.0),
        ambient=0.30,
        diffuse=0.78,
        specular=0.18,
        shininess=56.0,
    )

    device = f"cuda:{vis.cuda_device}"
    vertices = torch.as_tensor(vertices_np, device=device)
    faces = torch.as_tensor(faces_np, device=device)

    vis.add_mesh("bunny_mesh", vertices, faces, color=(0.20, 0.56, 0.90), offset=(-0.55, 0.0, 0.0))
    vis.add_points(
        "bunny_cloud",
        vertices,
        color=(0.92, 0.34, 0.24),
        point_size=args.point_size,
        offset=(0.55, 0.0, 0.0),
    )
    vis.set_view(target=(0.0, 0.0, 0.02), distance=2.4, yaw_deg=35.0, pitch_deg=18.0)

    print(
        f"{args.ply.name} | {vertices.shape[0]} points | "
        f"{faces.shape[0]} triangles"
    )

    try:
        while vis.is_open():
            vis.poll()
    finally:
        vis.destroy()
        glfw.terminate()


if __name__ == "__main__":
    main()
