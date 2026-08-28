import torch
from torch_gl_viewer import TorchGLViewer

vis = TorchGLViewer(
    "Tracking",
    width=1600,
    height=1000,
    background=(0.97, 0.97, 0.95),
    vsync=True,
    cuda_device=0,
)
vis.set_lighting(
    direction=(0.25, -0.45, -1.0),
    ambient=0.22,
    diffuse=0.82,
    specular=0.28,
    shininess=64.0,
)
device = f"cuda:{vis.cuda_device}"

# Example mesh
verts = torch.tensor(
    [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], device=device
)

faces = torch.tensor([[0, 1, 2]], device=device)

vis.add_mesh("live", verts, faces, color=(0.35, 0.72, 1.0))
vis.set_view(target=(0, 0, 0), distance=4)

try:
    while vis.is_open():
        # Your pipeline can update verts on CUDA here.
        # No .cpu(), no .numpy().
        verts[:, 2] = 0.15 * torch.sin(verts[:, 0] * 5.0)
        vis.update_mesh("live", vertices=verts)

        # Keeps mouse/keyboard/window responsive.
        vis.poll()
finally:
    vis.destroy()
