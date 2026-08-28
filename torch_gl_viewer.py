"""
torch_gl_viewer.py

Minimal high-performance interactive 3D viewer for CUDA PyTorch tensors.

Pipeline (CUDA inputs):
    torch.Tensor (CUDA)
        -> cudaMemcpyDeviceToDevice
        -> OpenGL VBO/EBO registered with CUDA
        -> OpenGL rasterizer
        -> screen

No GPU -> CPU -> GPU transfer for geometry.

Dependencies:
    pip install glfw PyOpenGL cuda-python torch numpy

Notes:
- Linux + NVIDIA CUDA/OpenGL is the primary target.
- The OpenGL context and the CUDA tensor must refer to the same physical GPU.
- Vertex positions: float32 [N, 3]
- Triangle indices: int32/int64 [M, 3] (int64 is converted to int32 on GPU)
"""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import glfw
import numpy as np
import torch
from OpenGL import GL

try:
    # cuda-python >= 12 style
    from cuda.bindings import runtime as cudart
except ImportError:
    # older cuda-python style
    from cuda import cudart


# ------------------------------- CUDA helpers -------------------------------


def _decode_cuda_text(x):
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    return str(x)


def _cuda_check(result, what="CUDA call"):
    """
    cuda-python functions usually return tuples:
      (error, ...)
    """
    err = result[0]
    # cudaSuccess is numerically 0 in CUDA runtime.
    if int(err) != 0:
        try:
            name = _decode_cuda_text(cudart.cudaGetErrorName(err)[1])
            msg = _decode_cuda_text(cudart.cudaGetErrorString(err)[1])
        except Exception:
            name, msg = str(err), ""
        raise RuntimeError(f"{what} failed: {name} {msg}")
    return result[1:]


def _cuda_device_name(device: int) -> str:
    try:
        (props,) = _cuda_check(
            cudart.cudaGetDeviceProperties(int(device)),
            "cudaGetDeviceProperties",
        )
        name = getattr(props, "name", b"")
        if isinstance(name, (bytes, bytearray)):
            return bytes(name).split(b"\0", 1)[0].decode(errors="replace")
        return str(name)
    except Exception:
        return "unknown CUDA device"


def _gl_string(name) -> str:
    value = GL.glGetString(name)
    if value is None:
        return "unknown"
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _current_gl_cuda_devices():
    if not hasattr(cudart, "cudaGLGetDevices"):
        return None
    count, devices = _cuda_check(
        cudart.cudaGLGetDevices(
            16,
            cudart.cudaGLDeviceList.cudaGLDeviceListAll,
        ),
        "cudaGLGetDevices",
    )
    return [int(devices[i]) for i in range(int(count))]


def _cuda_memcpy_d2d(dst_ptr: int, src_ptr: int, nbytes: int):
    if nbytes == 0:
        return
    _cuda_check(
        cudart.cudaMemcpy(
            dst_ptr,
            src_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        ),
        "cudaMemcpy D2D",
    )


def _register_gl_buffer(buffer_id: int):
    # Write-discard is ideal for dynamic geometry that is replaced each update.
    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    try:
        (resource,) = _cuda_check(
            cudart.cudaGraphicsGLRegisterBuffer(buffer_id, flags),
            "cudaGraphicsGLRegisterBuffer",
        )
    except RuntimeError as exc:
        vendor = _gl_string(GL.GL_VENDOR)
        renderer = _gl_string(GL.GL_RENDERER)
        raise RuntimeError(
            f"{exc}\n"
            f"OpenGL context: vendor={vendor!r}, renderer={renderer!r}. "
            "CUDA/OpenGL interop requires the GLFW OpenGL context and the "
            "PyTorch CUDA tensor to live on the same NVIDIA GPU. If this is "
            "an Optimus/PRIME machine, start the demo with prime-run or with "
            "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia."
        ) from exc
    return resource


def _unregister_gl_buffer(resource):
    if resource is not None:
        _cuda_check(
            cudart.cudaGraphicsUnregisterResource(resource),
            "cudaGraphicsUnregisterResource",
        )


def _cuda_graphics_map_resource(resource):
    try:
        return _cuda_check(
            cudart.cudaGraphicsMapResources(1, resource, 0),
            "cudaGraphicsMapResources",
        )
    except TypeError:
        return _cuda_check(
            cudart.cudaGraphicsMapResources(1, [resource], 0),
            "cudaGraphicsMapResources",
        )


def _cuda_graphics_unmap_resource(resource):
    try:
        return _cuda_check(
            cudart.cudaGraphicsUnmapResources(1, resource, 0),
            "cudaGraphicsUnmapResources",
        )
    except TypeError:
        return _cuda_check(
            cudart.cudaGraphicsUnmapResources(1, [resource], 0),
            "cudaGraphicsUnmapResources",
        )


def _copy_cuda_tensor_to_gl_buffer(resource, tensor: torch.Tensor):
    """Map GL buffer in CUDA and copy tensor device-to-device."""
    if tensor.numel() == 0:
        return
    if resource is None:
        raise RuntimeError("Cannot upload a non-empty CUDA tensor to an unregistered GL buffer")
    _cuda_graphics_map_resource(resource)
    try:
        ptr, size = _cuda_check(
            cudart.cudaGraphicsResourceGetMappedPointer(resource),
            "cudaGraphicsResourceGetMappedPointer",
        )
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > int(size):
            raise RuntimeError(
                f"OpenGL buffer too small: need {nbytes} bytes, mapped size={size}"
            )
        _cuda_memcpy_d2d(int(ptr), int(tensor.data_ptr()), nbytes)
    finally:
        _cuda_graphics_unmap_resource(resource)


# ------------------------------- math helpers -------------------------------


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def _look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = _normalize(target - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-8)
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


# ------------------------------- GL helpers ---------------------------------


def _compile_shader(src: str, shader_type):
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, src)
    GL.glCompileShader(shader)
    ok = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not ok:
        log = GL.glGetShaderInfoLog(shader).decode(errors="replace")
        GL.glDeleteShader(shader)
        raise RuntimeError(log)
    return shader


def _make_program(vs: str, fs: str):
    p = GL.glCreateProgram()
    a = _compile_shader(vs, GL.GL_VERTEX_SHADER)
    b = _compile_shader(fs, GL.GL_FRAGMENT_SHADER)
    GL.glAttachShader(p, a)
    GL.glAttachShader(p, b)
    GL.glLinkProgram(p)
    GL.glDeleteShader(a)
    GL.glDeleteShader(b)
    ok = GL.glGetProgramiv(p, GL.GL_LINK_STATUS)
    if not ok:
        log = GL.glGetProgramInfoLog(p).decode(errors="replace")
        GL.glDeleteProgram(p)
        raise RuntimeError(log)
    return p


def _uniform_location(cache: dict, program, name: str):
    key = (int(program), name)
    loc = cache.get(key)
    if loc is None:
        loc = GL.glGetUniformLocation(program, name)
        cache[key] = loc
    return loc


MESH_VS = r"""
#version 330 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_offset;
uniform vec3 u_color;
uniform bool u_use_vertex_color;

out vec3 v_world_pos;
out vec3 v_color;

void main() {
    vec3 p = in_pos + u_offset;
    v_world_pos = p;
    v_color = u_use_vertex_color ? in_color : u_color;
    gl_Position = u_proj * u_view * vec4(p, 1.0);
}
"""

# Flat-ish lighting without a normal buffer:
# derive a face normal from screen-space derivatives of world position.
MESH_FS = r"""
#version 330 core
in vec3 v_world_pos;
in vec3 v_color;
out vec4 out_color;

uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_view_pos;
uniform float u_ambient_strength;
uniform float u_diffuse_strength;
uniform float u_specular_strength;
uniform float u_shininess;

void main() {
    vec3 dx = dFdx(v_world_pos);
    vec3 dy = dFdy(v_world_pos);
    vec3 N = normalize(cross(dx, dy));
    if (!gl_FrontFacing) N = -N;

    vec3 L = normalize(-u_light_dir);
    vec3 V = normalize(u_view_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float ndotl = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), u_shininess);
    vec3 ambient_term = u_ambient_strength * v_color;
    vec3 diffuse_term = u_diffuse_strength * ndotl * v_color;
    vec3 specular_term = u_specular_strength * specular * u_light_color;
    out_color = vec4(ambient_term + diffuse_term + specular_term, 1.0);
}
"""

POINT_VS = r"""
#version 330 core
layout(location = 0) in vec3 in_pos;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_offset;
uniform float u_point_size;

void main() {
    vec3 p = in_pos + u_offset;
    gl_Position = u_proj * u_view * vec4(p, 1.0);
    gl_PointSize = u_point_size;
}
"""

POINT_FS = r"""
#version 330 core
out vec4 out_color;
uniform vec3 u_color;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    if (dot(d, d) > 1.0) discard;
    out_color = vec4(u_color, 1.0);
}
"""


TEXT_VS = r"""
#version 330 core
layout(location = 0) in vec2 in_pos;

uniform vec2 u_screen;
uniform vec2 u_offset;

void main() {
    vec2 p = ((in_pos + u_offset) / u_screen) * 2.0 - 1.0;
    gl_Position = vec4(p.x, p.y, 0.0, 1.0);
}
"""


TEXT_FS = r"""
#version 330 core
out vec4 out_color;
uniform vec4 u_color;

void main() {
    out_color = u_color;
}
"""


_FONT_5X7 = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10011", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "|": ("00100", "00100", "00100", "00100", "00100", "00100", "00100"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
}


@dataclass
class _GLBuffer:
    target: int
    gl_id: int = 0
    cuda_resource: object = None
    capacity_bytes: int = 0

    def __post_init__(self):
        self.gl_id = GL.glGenBuffers(1)

    def ensure_capacity(self, nbytes: int):
        if nbytes <= self.capacity_bytes:
            return

        if self.cuda_resource is not None:
            _unregister_gl_buffer(self.cuda_resource)
            self.cuda_resource = None

        # Geometric growth reduces expensive reallocations/registrations.
        new_capacity = max(nbytes, max(256, self.capacity_bytes * 2))

        GL.glBindBuffer(self.target, self.gl_id)
        GL.glBufferData(self.target, new_capacity, None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(self.target, 0)

        self.capacity_bytes = new_capacity
        self.cuda_resource = _register_gl_buffer(self.gl_id)

    def upload(self, x: torch.Tensor):
        nbytes = x.numel() * x.element_size()
        self.ensure_capacity(nbytes)
        _copy_cuda_tensor_to_gl_buffer(self.cuda_resource, x)

    def destroy(self):
        if self.cuda_resource is not None:
            _unregister_gl_buffer(self.cuda_resource)
            self.cuda_resource = None
        if self.gl_id:
            GL.glDeleteBuffers(1, [self.gl_id])
            self.gl_id = 0


class _MeshGPU:
    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = _GLBuffer(GL.GL_ARRAY_BUFFER)
        self.cbo = _GLBuffer(GL.GL_ARRAY_BUFFER)
        self.ebo = _GLBuffer(GL.GL_ELEMENT_ARRAY_BUFFER)
        self.n_vertices = 0
        self.n_indices = 0
        self.color = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        self.offset = np.zeros(3, dtype=np.float32)
        self.has_vertex_colors = False
        self.visible = True

    def bind_layout(self):
        GL.glBindVertexArray(self.vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo.gl_id)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 3 * 4, ctypes.c_void_p(0)
        )

        if self.has_vertex_colors:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo.gl_id)
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(
                1, 3, GL.GL_FLOAT, GL.GL_FALSE, 3 * 4, ctypes.c_void_p(0)
            )
        else:
            GL.glDisableVertexAttribArray(1)
            GL.glVertexAttrib3f(1, 1.0, 1.0, 1.0)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo.gl_id)
        GL.glBindVertexArray(0)

    def destroy(self):
        self.vbo.destroy()
        self.cbo.destroy()
        self.ebo.destroy()
        GL.glDeleteVertexArrays(1, [self.vao])


class _PointsGPU:
    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = _GLBuffer(GL.GL_ARRAY_BUFFER)
        self.n_points = 0
        self.color = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.offset = np.zeros(3, dtype=np.float32)
        self.point_size = 3.0
        self.visible = True

    def bind_layout(self):
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo.gl_id)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 3 * 4, ctypes.c_void_p(0)
        )
        GL.glBindVertexArray(0)

    def destroy(self):
        self.vbo.destroy()
        GL.glDeleteVertexArrays(1, [self.vao])


@dataclass
class _Lighting:
    direction: np.ndarray
    color: np.ndarray
    ambient: float = 0.25
    diffuse: float = 0.80
    specular: float = 0.20
    shininess: float = 48.0


class _TextOverlay:
    def __init__(self):
        self.program = _make_program(TEXT_VS, TEXT_FS)
        self._uniform_locations = {}
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.text = None
        self.vertices = np.zeros((0, 2), dtype=np.float32)
        self.width = 0
        self.height = 0

    def set_text(self, text: str, scale: int = 2):
        text = text.upper()
        if text == self.text:
            return
        self.text = text

        x = 0
        y = 0
        quads = []
        step = 6 * scale
        self.height = 7 * scale

        for ch in text:
            if ch == " ":
                x += step
                continue
            glyph = _FONT_5X7.get(ch)
            if glyph is None:
                x += step
                continue
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit != "1":
                        continue
                    x0 = x + col * scale
                    y0 = y + (6 - row) * scale
                    x1 = x0 + scale
                    y1 = y0 + scale
                    quads.extend(
                        [
                            (x0, y0),
                            (x1, y0),
                            (x1, y1),
                            (x0, y0),
                            (x1, y1),
                            (x0, y1),
                        ]
                    )
            x += step

        self.width = max(x - scale, 0)
        self.vertices = np.asarray(quads, dtype=np.float32)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self.vertices.nbytes,
            self.vertices,
            GL.GL_STATIC_DRAW,
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 2, GL.GL_FLOAT, GL.GL_FALSE, 2 * 4, ctypes.c_void_p(0)
        )
        GL.glBindVertexArray(0)

    def draw(
        self, screen_w: int, screen_h: int, x: int, y: int, color=(0.0, 0.0, 0.0, 0.88)
    ):
        if self.vertices.size == 0:
            return
        GL.glUseProgram(self.program)
        GL.glUniform2f(
            _uniform_location(self._uniform_locations, self.program, "u_screen"),
            screen_w,
            screen_h,
        )
        GL.glUniform2f(
            _uniform_location(self._uniform_locations, self.program, "u_offset"), x, y
        )
        GL.glUniform4f(
            _uniform_location(self._uniform_locations, self.program, "u_color"), *color
        )

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glBindVertexArray(self.vao)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])

        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def destroy(self):
        if self.vbo:
            GL.glDeleteBuffers(1, [self.vbo])
            self.vbo = 0
        if self.vao:
            GL.glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
        if self.program:
            GL.glDeleteProgram(self.program)
            self.program = 0


class TorchGLViewer:
    """
    Small Open3D-like viewer for CUDA torch tensors.

    Typical usage:

        vis = TorchGLViewer("Tracking")
        vis.add_mesh("smpl", verts, faces, color=(1, .5, 0))

        while vis.is_open():
            vis.update_mesh("smpl", vertices=new_verts)
            vis.poll()

        vis.destroy()

    Interactions:
        Left drag   : orbit
        Right drag  : pan
        Wheel       : zoom
        R           : reset camera
        Esc         : close
    """

    def __init__(
        self,
        title="Torch CUDA Viewer",
        width=1600,
        height=1000,
        background=(0.06, 0.07, 0.08),
        vsync=False,
        cuda_device=0,
        show_fps=True,
        show_controls=True,
        light_dir=(0.4, -0.6, -1.0),
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is required.")

        self.cuda_device = int(cuda_device)

        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if vsync else 0)

        self._validate_cuda_gl_device()
        torch.cuda.set_device(self.cuda_device)

        self.title = str(title)
        self.background = tuple(float(x) for x in background)
        self.show_fps = bool(show_fps)
        self.show_controls = bool(show_controls)
        self._fps_frames = 0
        self._fps_value = 0.0
        self._fps_last_time = time.perf_counter()
        self._graphics_command_started_at = None
        self._render_ms = 0.0
        self._render_ms_ema = 0.0
        self._upload_ms_frame = 0.0
        self._upload_ms_ema = 0.0
        self._draw_ms_frame = 0.0
        self._draw_ms_ema = 0.0
        self._present_ms_frame = 0.0
        self._present_ms_ema = 0.0
        self.lighting = _Lighting(
            direction=_normalize(light_dir).astype(np.float32),
            color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        GL.glDisable(GL.GL_CULL_FACE)

        self.mesh_program = _make_program(MESH_VS, MESH_FS)
        self.point_program = _make_program(POINT_VS, POINT_FS)
        self._uniform_locations = {}
        self.text_overlay = _TextOverlay()
        self.text_overlay.set_text(
            "LMB ORBIT | RMB PAN | WHEEL ZOOM | R RESET | ESC CLOSE"
        )
        self.triangle_overlay = _TextOverlay()
        self._triangle_overlay_count = None

        self.meshes: Dict[str, _MeshGPU] = {}
        self.points: Dict[str, _PointsGPU] = {}

        self.target = np.array([0.0, 0.0, 0.0], np.float32)
        self.distance = 3.0
        self.yaw = math.radians(45.0)
        self.pitch = math.radians(20.0)
        self.fovy = 45.0

        self._mouse_last = None
        self._mouse_button = None

        glfw.set_window_user_pointer(self.window, self)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_cb)
        glfw.set_cursor_pos_callback(self.window, self._cursor_cb)
        glfw.set_scroll_callback(self.window, self._scroll_cb)
        glfw.set_key_callback(self.window, self._key_cb)

    # ------------------------------- API ----------------------------------

    def add_mesh(
        self,
        name: str,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        color=(0.8, 0.8, 0.8),
        vertex_colors: Optional[torch.Tensor] = None,
        offset=(0.0, 0.0, 0.0),
    ):
        if name in self.meshes:
            raise KeyError(f"Mesh {name!r} already exists")

        self._begin_graphics_command()
        m = _MeshGPU()
        self.meshes[name] = m
        m.color = np.asarray(color, dtype=np.float32)
        m.offset = np.asarray(offset, dtype=np.float32)
        self.update_mesh(name, vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        return name

    def update_mesh(
        self,
        name: str,
        vertices: Optional[torch.Tensor] = None,
        faces: Optional[torch.Tensor] = None,
        color: Optional[Sequence[float]] = None,
        vertex_colors: Optional[torch.Tensor] = None,
        offset: Optional[Sequence[float]] = None,
    ):
        self._begin_graphics_command()
        m = self.meshes[name]

        if vertices is not None:
            v = self._positions(vertices)
            upload_started_at = time.perf_counter()
            m.vbo.upload(v)
            self._upload_ms_frame += (time.perf_counter() - upload_started_at) * 1000.0
            m.n_vertices = v.shape[0]

        if vertex_colors is not None:
            c = self._colors(vertex_colors)
            if c.shape[0] != m.n_vertices:
                raise ValueError(
                    f"Expected {m.n_vertices} vertex colors, got {c.shape[0]}"
                )
            upload_started_at = time.perf_counter()
            m.cbo.upload(c)
            self._upload_ms_frame += (time.perf_counter() - upload_started_at) * 1000.0
            m.has_vertex_colors = c.shape[0] > 0
        elif vertices is not None:
            m.has_vertex_colors = False

        if faces is not None:
            f = self._faces(faces)
            upload_started_at = time.perf_counter()
            m.ebo.upload(f)
            self._upload_ms_frame += (time.perf_counter() - upload_started_at) * 1000.0
            m.n_indices = f.numel()

        if color is not None:
            m.color = np.asarray(color, dtype=np.float32)
        if offset is not None:
            m.offset = np.asarray(offset, dtype=np.float32)

        # Buffer IDs stay constant unless buffer capacity had to grow.
        # Re-binding is cheap and ensures VAO state is correct.
        m.bind_layout()

    def add_points(
        self,
        name: str,
        points: torch.Tensor,
        color=(0.9, 0.9, 0.9),
        point_size=3.0,
        offset=(0.0, 0.0, 0.0),
    ):
        if name in self.points:
            raise KeyError(f"Point cloud {name!r} already exists")

        self._begin_graphics_command()
        p = _PointsGPU()
        self.points[name] = p
        p.color = np.asarray(color, dtype=np.float32)
        p.point_size = float(point_size)
        p.offset = np.asarray(offset, dtype=np.float32)
        self.update_points(name, points=points)
        return name

    def update_points(
        self,
        name: str,
        points: Optional[torch.Tensor] = None,
        color: Optional[Sequence[float]] = None,
        point_size: Optional[float] = None,
        offset: Optional[Sequence[float]] = None,
    ):
        self._begin_graphics_command()
        p = self.points[name]

        if points is not None:
            x = self._positions(points)
            upload_started_at = time.perf_counter()
            p.vbo.upload(x)
            self._upload_ms_frame += (time.perf_counter() - upload_started_at) * 1000.0
            p.n_points = x.shape[0]

        if color is not None:
            p.color = np.asarray(color, dtype=np.float32)
        if point_size is not None:
            p.point_size = float(point_size)
        if offset is not None:
            p.offset = np.asarray(offset, dtype=np.float32)

        p.bind_layout()

    def remove(self, name: str):
        self._begin_graphics_command()
        if name in self.meshes:
            self.meshes.pop(name).destroy()
        if name in self.points:
            self.points.pop(name).destroy()

    def set_visible(self, name: str, visible: bool):
        self._begin_graphics_command()
        if name in self.meshes:
            self.meshes[name].visible = bool(visible)
        elif name in self.points:
            self.points[name].visible = bool(visible)
        else:
            raise KeyError(name)

    def set_mesh_color(self, name: str, color: Sequence[float]):
        self._begin_graphics_command()
        if name not in self.meshes:
            raise KeyError(name)
        self.meshes[name].color = np.asarray(color, dtype=np.float32)

    def set_lighting(
        self,
        direction: Optional[Sequence[float]] = None,
        color: Optional[Sequence[float]] = None,
        ambient: Optional[float] = None,
        diffuse: Optional[float] = None,
        specular: Optional[float] = None,
        shininess: Optional[float] = None,
    ):
        self._begin_graphics_command()
        if direction is not None:
            self.lighting.direction = _normalize(direction).astype(np.float32)
        if color is not None:
            self.lighting.color = np.asarray(color, dtype=np.float32)
        if ambient is not None:
            self.lighting.ambient = float(ambient)
        if diffuse is not None:
            self.lighting.diffuse = float(diffuse)
        if specular is not None:
            self.lighting.specular = float(specular)
        if shininess is not None:
            self.lighting.shininess = max(float(shininess), 1.0)

    def set_view(
        self,
        target=None,
        distance=None,
        yaw_deg=None,
        pitch_deg=None,
        fovy=None,
    ):
        self._begin_graphics_command()
        if target is not None:
            self.target = np.asarray(target, np.float32)
        if distance is not None:
            self.distance = max(float(distance), 1e-4)
        if yaw_deg is not None:
            self.yaw = math.radians(float(yaw_deg))
        if pitch_deg is not None:
            self.pitch = math.radians(float(pitch_deg))
        if fovy is not None:
            self.fovy = float(fovy)

    def render_stats(self):
        return {
            "fps": self._fps_value,
            "upload_ms": self._upload_ms_ema,
            "draw_ms": self._draw_ms_ema,
            "present_ms": self._present_ms_ema,
            "view_ms": self._upload_ms_ema + self._draw_ms_ema,
            "total_ms": self._render_ms_ema,
        }

    def is_open(self):
        return self.window is not None and not glfw.window_should_close(self.window)

    def poll(self):
        if not self.is_open():
            return False

        poll_started_at = time.perf_counter()
        render_started_at = self._graphics_command_started_at or poll_started_at
        self._draw_ms_frame = 0.0
        self._present_ms_frame = 0.0

        glfw.make_context_current(self.window)
        glfw.poll_events()

        w, h = glfw.get_framebuffer_size(self.window)
        if w <= 0 or h <= 0:
            return True

        GL.glViewport(0, 0, w, h)
        GL.glClearColor(*self.background, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        eye = self._camera_eye()
        view, proj = self._camera_matrices(w, h, eye=eye)
        draw_started_at = time.perf_counter()

        # meshes
        GL.glUseProgram(self.mesh_program)
        self._set_mat4(self.mesh_program, "u_view", view)
        self._set_mat4(self.mesh_program, "u_proj", proj)
        GL.glUniform3fv(
            self._uniform_location(self.mesh_program, "u_light_dir"),
            1,
            self.lighting.direction,
        )
        GL.glUniform3fv(
            self._uniform_location(self.mesh_program, "u_light_color"),
            1,
            self.lighting.color,
        )
        GL.glUniform3fv(self._uniform_location(self.mesh_program, "u_view_pos"), 1, eye)
        GL.glUniform1f(
            self._uniform_location(self.mesh_program, "u_ambient_strength"),
            self.lighting.ambient,
        )
        GL.glUniform1f(
            self._uniform_location(self.mesh_program, "u_diffuse_strength"),
            self.lighting.diffuse,
        )
        GL.glUniform1f(
            self._uniform_location(self.mesh_program, "u_specular_strength"),
            self.lighting.specular,
        )
        GL.glUniform1f(
            self._uniform_location(self.mesh_program, "u_shininess"),
            self.lighting.shininess,
        )

        for m in self.meshes.values():
            if not m.visible or m.n_indices == 0:
                continue
            GL.glUniform3fv(
                self._uniform_location(self.mesh_program, "u_color"), 1, m.color
            )
            GL.glUniform1i(
                self._uniform_location(self.mesh_program, "u_use_vertex_color"),
                int(m.has_vertex_colors),
            )
            GL.glUniform3fv(
                self._uniform_location(self.mesh_program, "u_offset"), 1, m.offset
            )
            GL.glBindVertexArray(m.vao)
            GL.glDrawElements(
                GL.GL_TRIANGLES,
                m.n_indices,
                GL.GL_UNSIGNED_INT,
                ctypes.c_void_p(0),
            )

        # points
        GL.glUseProgram(self.point_program)
        self._set_mat4(self.point_program, "u_view", view)
        self._set_mat4(self.point_program, "u_proj", proj)

        for p in self.points.values():
            if not p.visible or p.n_points == 0:
                continue
            GL.glUniform3fv(
                self._uniform_location(self.point_program, "u_color"), 1, p.color
            )
            GL.glUniform3fv(
                self._uniform_location(self.point_program, "u_offset"), 1, p.offset
            )
            GL.glUniform1f(
                self._uniform_location(self.point_program, "u_point_size"), p.point_size
            )
            GL.glBindVertexArray(p.vao)
            GL.glDrawArrays(GL.GL_POINTS, 0, p.n_points)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        if self.show_controls:
            text_color = (0.0, 0.0, 0.0, 0.88)
            self.text_overlay.draw(w, h, 12, 12, color=text_color)
            triangles = self._visible_triangle_count()
            if triangles != self._triangle_overlay_count:
                self.triangle_overlay.set_text(f"TRIANGLES {triangles}")
                self._triangle_overlay_count = triangles
            x = max(12, w - self.triangle_overlay.width - 12)
            self.triangle_overlay.draw(w, h, x, 12, color=text_color)

        self._draw_ms_frame = (time.perf_counter() - draw_started_at) * 1000.0
        present_started_at = time.perf_counter()
        glfw.swap_buffers(self.window)
        self._present_ms_frame = (time.perf_counter() - present_started_at) * 1000.0
        self._finish_render_timing(render_started_at)
        self._graphics_command_started_at = None
        self._update_title_stats()
        self._upload_ms_frame = 0.0
        return True

    def destroy(self):
        if self.window is None:
            return
        glfw.make_context_current(self.window)

        for m in list(self.meshes.values()):
            m.destroy()
        for p in list(self.points.values()):
            p.destroy()
        self.meshes.clear()
        self.points.clear()

        GL.glDeleteProgram(self.mesh_program)
        GL.glDeleteProgram(self.point_program)
        self.text_overlay.destroy()
        self.triangle_overlay.destroy()

        glfw.destroy_window(self.window)
        self.window = None
        glfw.terminate()

    # ----------------------------- internals -------------------------------

    def _validate_cuda_gl_device(self):
        vendor = _gl_string(GL.GL_VENDOR)
        renderer = _gl_string(GL.GL_RENDERER)

        try:
            gl_cuda_devices = _current_gl_cuda_devices()
        except RuntimeError as exc:
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()
            raise RuntimeError(
                f"CUDA/OpenGL interop is not available for this OpenGL context.\n"
                f"OpenGL context: vendor={vendor!r}, renderer={renderer!r}.\n"
                f"Requested CUDA device: cuda:{self.cuda_device} "
                f"({_cuda_device_name(self.cuda_device)}).\n"
                "On hybrid graphics systems, run the process on the NVIDIA GPU, "
                "for example with `prime-run python3 ...` or "
                "`__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python3 ...`."
            ) from exc

        if gl_cuda_devices is None:
            return

        if self.cuda_device not in gl_cuda_devices:
            names = ", ".join(
                f"cuda:{d} ({_cuda_device_name(d)})" for d in gl_cuda_devices
            )
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()
            raise RuntimeError(
                "CUDA/OpenGL device mismatch.\n"
                f"OpenGL context: vendor={vendor!r}, renderer={renderer!r}.\n"
                f"OpenGL-compatible CUDA devices: {names or 'none'}.\n"
                f"Requested CUDA device: cuda:{self.cuda_device} "
                f"({_cuda_device_name(self.cuda_device)}).\n"
                "Pass the matching `cuda_device=` to TorchGLViewer and create "
                "input tensors on the same CUDA device."
            )

    def _positions(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected torch.Tensor")
        if not x.is_cuda:
            raise ValueError("Geometry must be a CUDA tensor")
        if x.device.index not in (None, self.cuda_device):
            raise ValueError(
                f"Tensor is on CUDA:{x.device.index}, viewer uses CUDA:{self.cuda_device}"
            )
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"Expected [N,3], got {tuple(x.shape)}")
        # Conversion/contiguity stays on GPU.
        return x.detach().to(dtype=torch.float32).contiguous()

    def _faces(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected torch.Tensor")
        if not x.is_cuda:
            raise ValueError("Faces must be a CUDA tensor")
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"Expected [M,3], got {tuple(x.shape)}")
        # OpenGL indexed draw uses uint32 here.
        return x.detach().to(dtype=torch.int32).contiguous()

    def _colors(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected torch.Tensor")
        if not x.is_cuda:
            raise ValueError("Vertex colors must be a CUDA tensor")
        if x.device.index not in (None, self.cuda_device):
            raise ValueError(
                f"Tensor is on CUDA:{x.device.index}, viewer uses CUDA:{self.cuda_device}"
            )
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"Expected [N,3], got {tuple(x.shape)}")
        return x.detach().to(dtype=torch.float32).clamp(0.0, 1.0).contiguous()

    def _camera_eye(self):
        eye, _forward, _right, _up = self._camera_basis()
        return eye

    def _camera_basis(self):
        cp = math.cos(self.pitch)
        direction = np.array(
            [
                cp * math.cos(self.yaw),
                cp * math.sin(self.yaw),
                math.sin(self.pitch),
            ],
            dtype=np.float32,
        )
        eye = self.target + direction * self.distance
        forward = _normalize(self.target - eye)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if cp < 0.0:
            world_up = -world_up
        right = _normalize(np.cross(forward, world_up))
        if np.linalg.norm(right) < 1e-6:
            yaw_right = np.array(
                [-math.sin(self.yaw), math.cos(self.yaw), 0.0],
                dtype=np.float32,
            )
            right = _normalize(yaw_right)
        up = _normalize(np.cross(right, forward))
        return eye, forward, right, up

    def _camera_matrices(self, w, h, eye=None):
        if eye is None:
            eye, _forward, _right, up = self._camera_basis()
        else:
            _basis_eye, _forward, _right, up = self._camera_basis()
        proj = _perspective(
            self.fovy,
            w / max(h, 1),
            max(1e-4, self.distance * 0.001),
            max(1000.0, self.distance * 100.0),
        )
        return _look_at(eye, self.target, up), proj

    def _begin_graphics_command(self):
        if self._graphics_command_started_at is None:
            self._graphics_command_started_at = time.perf_counter()

    def _visible_triangle_count(self):
        return sum(
            int(m.n_indices // 3)
            for m in self.meshes.values()
            if m.visible and m.n_indices > 0
        )

    def _finish_render_timing(self, render_started_at):
        self._render_ms = (time.perf_counter() - render_started_at) * 1000.0
        self._render_ms_ema = self._ema(self._render_ms_ema, self._render_ms)
        self._upload_ms_ema = self._ema(self._upload_ms_ema, self._upload_ms_frame)
        self._draw_ms_ema = self._ema(self._draw_ms_ema, self._draw_ms_frame)
        self._present_ms_ema = self._ema(self._present_ms_ema, self._present_ms_frame)

    @staticmethod
    def _ema(previous, value, alpha=0.15):
        if previous <= 0.0:
            return value
        return (1.0 - alpha) * previous + alpha * value

    def _update_title_stats(self):
        self._fps_frames += 1
        now = time.perf_counter()
        dt = now - self._fps_last_time
        if dt < 0.25:
            return
        self._fps_value = self._fps_frames / dt
        self._fps_frames = 0
        self._fps_last_time = now
        if self.show_fps:
            view_ms = self._upload_ms_ema + self._draw_ms_ema
            glfw.set_window_title(
                self.window,
                f"{self.title} | {self._fps_value:6.1f} FPS | "
                f"upload {self._upload_ms_ema:5.2f} ms | "
                f"draw {self._draw_ms_ema:5.2f} ms | "
                f"present {self._present_ms_ema:5.2f} ms | "
                f"view {view_ms:5.2f} ms | "
                f"latency {self._render_ms_ema:5.2f} ms",
            )

    def _uniform_location(self, program, name: str):
        return _uniform_location(self._uniform_locations, program, name)

    def _set_mat4(self, program, name, m):
        loc = self._uniform_location(program, name)
        # numpy is row-major; transpose=True gives GL the intended matrix.
        GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, m)

    # ------------------------------ input ----------------------------------

    def _mouse_button_cb(self, window, button, action, mods):
        if action == glfw.PRESS:
            self._mouse_button = button
            self._mouse_last = np.array(glfw.get_cursor_pos(window), np.float32)
        elif action == glfw.RELEASE:
            self._mouse_button = None
            self._mouse_last = None

    def _cursor_cb(self, window, x, y):
        if self._mouse_button is None or self._mouse_last is None:
            return

        now = np.array([x, y], np.float32)
        d = now - self._mouse_last
        self._mouse_last = now

        if self._mouse_button == glfw.MOUSE_BUTTON_LEFT:
            self.yaw -= float(d[0]) * 0.006
            self.pitch += float(d[1]) * 0.006
            self.pitch = math.atan2(math.sin(self.pitch), math.cos(self.pitch))

        elif self._mouse_button == glfw.MOUSE_BUTTON_RIGHT:
            _eye, _forward, right, up = self._camera_basis()
            scale = self.distance * 0.0015
            self.target += (-right * d[0] + up * d[1]) * scale

    def _scroll_cb(self, window, dx, dy):
        self.distance *= math.exp(-0.005 * float(dy))
        self.distance = max(self.distance, 1e-4)

    def _key_cb(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            self.target[:] = 0
            self.distance = 3.0
            self.yaw = math.radians(45.0)
            self.pitch = math.radians(20.0)


# --------------------------------- demo -------------------------------------

if __name__ == "__main__":
    dev = "cuda:0"

    # Cube
    verts = torch.tensor(
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
        dtype=torch.float32,
        device=dev,
    )

    faces = torch.tensor(
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
        dtype=torch.int32,
        device=dev,
    )

    vis = TorchGLViewer("CUDA PyTorch Mesh Viewer", 1400, 900, vsync=True)
    vis.add_mesh("cube", verts, faces, color=(0.95, 0.55, 0.15))
    vis.set_view(target=(0, 0, 0), distance=5.0)

    # Animate completely on CUDA; only a D2D transfer enters the GL VBO.
    t = 0.0
    try:
        while vis.is_open():
            t += 0.02
            deformed = verts.clone()
            deformed[:, 2] += 0.15 * torch.sin(
                deformed[:, 0] * 3.0 + torch.tensor(t, device=dev)
            )
            vis.update_mesh("cube", vertices=deformed)
            vis.poll()
    finally:
        vis.destroy()
