"""ctypes loader for the compiled Mojo kernels."""

from __future__ import annotations

import atexit
import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOPYVISTA_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyvista.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpv_parallel_init": ([], I),
    "mpv_transform_points": ([I, I, I, I], None),
    "mpv_transform_vectors": ([I, I, I, I], None),
    "mpv_warp_scalar": ([I, I, I, I, I, F], None),
    "mpv_warp_vector": ([I, I, I, I, F], None),
    "mpv_elevation": ([I, I, I] + [F] * 8, None),
    "mpv_cell_areas": ([I, I, I, I, I], None),
    "mpv_cell_geometry": ([I, I, I, I, I, I, I], None),
    "mpv_point_normals": ([I, I, I, I, I, I], None),
    "mpv_triangulate": ([I, I, I], I),
    "mpv_cell_to_point": ([I, I, I, I, I, I], None),
    "mpv_build_point_adjacency": ([I, I, I, I, I], None),
    "mpv_point_to_cell": ([I, I, I, I, I, I], None),
}

_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first"
            )
        _lib = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
        parallel_device = _lib.mpv_parallel_init()
        release_device = _lib.KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice
        release_device.argtypes = [I]
        release_device.restype = None
        atexit.register(release_device, parallel_device)
    return _lib


def f64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "biuf":
        raise TypeError("values must be real numbers")
    if source.dtype.kind == "f" and source.dtype.itemsize > 8:
        raise TypeError(f"values cannot be narrowed from {source.dtype} to float64")
    if source.dtype.kind in "iu" and source.size:
        limit = 2**53
        if np.any(source > limit) or (
            source.dtype.kind == "i" and np.any(source < -limit)
        ):
            raise OverflowError("integers must be exactly representable as float64")
    if copy:
        return np.array(source, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(source, dtype=np.float64)


def i64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    return array.ctypes.data
