# mojo-pyvista

`mojo-pyvista` is a focused port of PyVista's polygon-mesh filters to Mojo. It
provides a small `PolyData` type whose covered methods use the same names and
core options as PyVista, with NumPy arrays at the Python boundary and compiled
Mojo for the array and geometry loops.

This is not a replacement for all of PyVista. It is intended for applications
that already have polygonal mesh arrays and need fast, dependency-light filter
execution.

## Covered subset

The following `PolyData` filters are implemented and parity-tested against
PyVista 0.48.4:

- `compute_normals`
- `triangulate`
- `transform`
- `warp_by_scalar`
- `warp_by_vector`
- `elevation`
- `cell_data_to_point_data`
- `point_data_to_cell_data`
- `compute_cell_sizes`
- `cell_centers`

The Python layer also covers packed-face construction, `point_data`,
`cell_data`, `field_data`, dictionary-style array access, active
scalar/vector selection, copying, bounds, centers, and `wrap()` for copying a
polygonal PyVista object.

The implementation supports polygon faces, including triangles, quads, and
convex n-gons. It does not support vertex, line, or triangle-strip cells;
volumetric and structured datasets; rendering; readers and writers; VTK
pipeline objects; boolean, clipping, contour, or connectivity filters;
categorical point-to-cell conversion; split vertices; automatic normal
orientation; or PyVista's exact concave n-gon triangulator. Default
orientation options are correct when input face winding is already
consistent.

## Install

The repository pins the tested Mojo nightly and installs PyVista for parity
tests and benchmarks:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-pyvista.so`. The activated Pixi
environment adds `python/` to `PYTHONPATH`.

## Usage

```python
import numpy as np
import mojopyvista as pv

points = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
mesh = pv.PolyData(points, faces=[4, 0, 1, 2, 3])
mesh["height"] = np.array([0.0, 0.2, 0.5, 0.1])
mesh.set_active_scalars("height", preference="point")

warped = mesh.warp_by_scalar(factor=2.0, normal=(0, 0, 1))
triangles = warped.triangulate()
areas = triangles.compute_cell_sizes()["Area"]

print(triangles.points)
print(areas)
```

## Benchmarks

These are real best-of-four timings from `pixi run bench`. A ratio above
`1.00x` means Mojo was faster. The benchmark includes Python wrapper and
allocation time, not just an isolated kernel.

Machine: Intel Xeon E5-2697 v4 at 2.30 GHz, 72 logical CPUs, Linux x86-64,
Python 3.13.14.

| operation | Mojo | PyVista/VTK | PyVista / Mojo |
|---|---:|---:|---:|
| transform, 640k points | 22.69 ms | 49.41 ms | 2.18x |
| warp_by_scalar, 640k points | 16.80 ms | 9.25 ms | 0.55x |
| warp_by_vector, 640k points | 12.57 ms | 8.54 ms | 0.68x |
| elevation, 640k points | 14.99 ms | 11.21 ms | 0.75x |
| compute_normals, 250k quads | 28.09 ms | 119.16 ms | 4.24x |
| triangulate, 250k quads | 7.48 ms | 60.38 ms | 8.07x |
| compute_cell_sizes, 250k quads | 5.36 ms | 31.02 ms | 5.78x |
| cell_data_to_point_data, 250k quads | 22.08 ms | 29.61 ms | 1.34x |
| point_data_to_cell_data, 251k points | 6.21 ms | 11.89 ms | 1.91x |

Results depend on mesh shape, CPU, allocator, and PyVista/VTK build. Run
`pixi run bench` on the target machine instead of treating these numbers as
universal.

No GPU path is included.

## How it works

Python owns every allocation. Points are contiguous row-major `float64`
arrays with shape `(n_points, 3)`. Faces use PyVista's legacy packed `int64`
layout: `[count, id0, ..., count, id0, ...]`. Scalar and vector arrays are
contiguous `float64` buffers while inside a kernel. Narrowing conversions from
wider floating-point types or integers that cannot be represented exactly are
rejected instead of passed across the ABI.

One Mojo compilation unit exports a fixed, non-parametric C ABI. `ctypes`
passes each NumPy buffer as an integer address, and Mojo reconstructs mutable
typed pointers using `AnyOrigin[mut=True]`. No Mojo allocation crosses the
FFI boundary. Before each call, Python validates dtype, dimensionality,
contiguity, topology bounds, and array lengths. The call is synchronous, and
Python retains references to every input and output buffer for its duration.
Filters that only replace points or add arrays share topology and unchanged
data arrays in their result, while public `copy()` remains deep by default.

Parity tests compare numerical arrays, connectivity, active-array behavior,
in-place behavior, and pass-through options directly with upstream PyVista.
Run them with `pixi run test`.
