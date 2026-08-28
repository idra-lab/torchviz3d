from __future__ import annotations

import argparse
import time

import glfw
import numpy as np
import torch

try:
    from .torch_gl_viewer import TorchGLViewer
except ImportError:
    from torch_gl_viewer import TorchGLViewer


BOX_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ],
    dtype=np.int32,
)


def box(center, size, pivot, swing):
    center = np.asarray(center, np.float32)
    size = 0.5 * np.asarray(size, np.float32)
    corners = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    verts = center + corners * size
    pivots = np.repeat(np.asarray(pivot, np.float32)[None], 8, axis=0)
    swings = np.full(8, swing, dtype=np.float32)
    return verts, BOX_FACES.copy(), pivots, swings


def avatar_mesh():
    parts = [
        ((0.0, 0.0, 1.05), (0.42, 0.24, 0.75), (0.0, 0.0, 1.05), 0.04),
        ((0.0, 0.0, 1.58), (0.30, 0.28, 0.30), (0.0, 0.0, 1.58), 0.02),
        ((-0.36, 0.0, 1.12), (0.16, 0.16, 0.65), (-0.25, 0.0, 1.38), 0.75),
        ((0.36, 0.0, 1.12), (0.16, 0.16, 0.65), (0.25, 0.0, 1.38), -0.75),
        ((-0.13, 0.0, 0.45), (0.17, 0.17, 0.82), (-0.13, 0.0, 0.82), -0.55),
        ((0.13, 0.0, 0.45), (0.17, 0.17, 0.82), (0.13, 0.0, 0.82), 0.55),
    ]
    verts, faces, pivots, swings = [], [], [], []
    for part in parts:
        v, f, p, s = box(*part)
        faces.append(f + len(verts) * 8)
        verts.append(v)
        pivots.append(p)
        swings.append(s)
    return np.vstack(verts), np.vstack(faces), np.vstack(pivots), np.concatenate(swings)


def grid_offsets(n, spacing, device):
    side = int(np.ceil(np.sqrt(n)))
    ij = torch.arange(n, device=device)
    x = (ij % side).float() - 0.5 * (side - 1)
    y = (ij // side).float() - 0.5 * (side - 1)
    return torch.stack([x * spacing, y * spacing, torch.zeros_like(x)], dim=1)


def expand_faces(faces, copies, verts_per_avatar):
    return np.vstack([faces + i * verts_per_avatar for i in range(copies)]).astype(
        np.int32
    )


def animate(base, pivots, swings, offsets, phases, t):
    rel = base[None] - pivots[None]
    angle = swings[None] * torch.sin(t + phases[:, None])
    c, s = torch.cos(angle), torch.sin(angle)
    x = rel[..., 0].expand_as(angle)
    y = rel[..., 1] * c - rel[..., 2] * s
    z = rel[..., 1] * s + rel[..., 2] * c
    verts = torch.stack([x, y, z], dim=-1) + pivots[None] + offsets[:, None]
    verts[..., 2] += 0.035 * torch.sin(t * 2.0 + phases)[:, None]
    return verts.reshape(-1, 3).contiguous()


def main():
    parser = argparse.ArgumentParser(description="Procedural CUDA/GL avatar army demo.")
    parser.add_argument("--avatars", type=int, default=1024)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--vsync", action="store_true")
    args = parser.parse_args()

    vis = TorchGLViewer(
        "Procedural Avatar Army",
        args.width,
        args.height,
        background=(1.0, 1.0, 1.0),
        vsync=args.vsync,
        cuda_device=args.cuda_device,
        show_fps=True,
        show_controls=True,
    )
    vis.set_lighting(ambient=0.28, diffuse=0.78, specular=0.12, shininess=48.0)

    device = f"cuda:{vis.cuda_device}"
    base_v, base_f, pivots, swings = avatar_mesh()
    base = torch.as_tensor(base_v, device=device)
    pivots = torch.as_tensor(pivots, device=device)
    swings = torch.as_tensor(swings, device=device)
    offsets = grid_offsets(args.avatars, args.spacing, device)
    phases = torch.linspace(0.0, 6.28, args.avatars, device=device)
    faces = torch.as_tensor(
        expand_faces(base_f, args.avatars, base_v.shape[0]), device=device
    )

    verts = animate(base, pivots, swings, offsets, phases, 0.0)
    vis.add_mesh("army", verts, faces, color=(0.18, 0.56, 0.92))
    side = int(np.ceil(np.sqrt(args.avatars)))
    vis.set_view(
        target=(0, 0, 0.8),
        distance=max(5.0, side * args.spacing * 1.15),
        yaw_deg=42,
        pitch_deg=28,
    )

    print(
        f"{args.avatars} avatars | {verts.shape[0]} vertices | {faces.shape[0]} faces"
    )
    t0 = time.perf_counter()
    try:
        while vis.is_open():
            t = (time.perf_counter() - t0) * 3.0
            verts = animate(base, pivots, swings, offsets, phases, t)
            vis.update_mesh("army", vertices=verts)
            vis.poll()
    finally:
        vis.destroy()
        glfw.terminate()


if __name__ == "__main__":
    main()
