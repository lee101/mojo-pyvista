"""Locked benchmark of Mojo kernels against PyVista/VTK on identical meshes."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pyvista as pv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojopyvista as mpv  # noqa: E402


def grid(nx: int, ny: int):
    x, y = np.meshgrid(
        np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="xy"
    )
    z = 0.05 * np.sin(3 * x) * np.cos(4 * y)
    points = np.ascontiguousarray(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
    rows = np.arange(ny - 1)[:, None]
    cols = np.arange(nx - 1)[None, :]
    p0 = (rows * nx + cols).ravel()
    cells = np.column_stack((p0, p0 + 1, p0 + nx + 1, p0 + nx))
    packed = np.empty((len(cells), 5), dtype=np.int64)
    packed[:, 0] = 4
    packed[:, 1:] = cells
    faces = packed.ravel()
    return mpv.PolyData(points, faces), pv.PolyData(points, faces)


def best_time(fn, repeat: int = 4) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


CASES = []


def case(name):
    def decorate(builder):
        CASES.append((name, builder))
        return builder

    return decorate


@case("transform, 640k points")
def transform_case():
    ours, upstream = grid(800, 800)
    matrix = np.array(
        [[0.8, -0.3, 0, 2], [0.3, 0.8, 0, -1], [0, 0, 1.2, 3], [0, 0, 0, 1]]
    )
    return (
        lambda: ours.transform(matrix, inplace=False),
        lambda: upstream.transform(matrix, inplace=False),
    )


@case("warp_by_scalar, 640k points")
def warp_scalar_case():
    ours, upstream = grid(800, 800)
    values = np.ascontiguousarray(np.sin(ours.points[:, 0] * 5))
    normals = np.tile([0.0, 0.0, 1.0], (ours.n_points, 1))
    for mesh in (ours, upstream):
        mesh.point_data["s"] = values
        mesh.point_data["Normals"] = normals
        mesh.set_active_scalars("s", preference="point")
    return (
        lambda: ours.warp_by_scalar(factor=0.5),
        lambda: upstream.warp_by_scalar(factor=0.5),
    )


@case("warp_by_vector, 640k points")
def warp_vector_case():
    ours, upstream = grid(800, 800)
    vectors = np.ascontiguousarray(
        np.column_stack(
            (
                np.sin(ours.points[:, 1]),
                np.cos(ours.points[:, 0]),
                np.full(ours.n_points, 0.1),
            )
        )
    )
    for mesh in (ours, upstream):
        mesh.point_data["v"] = vectors
        mesh.set_active_vectors("v")
    return (
        lambda: ours.warp_by_vector(factor=0.5),
        lambda: upstream.warp_by_vector(factor=0.5),
    )


@case("elevation, 640k points")
def elevation_case():
    ours, upstream = grid(800, 800)
    args = dict(
        low_point=(-1, 0, 0),
        high_point=(1, 0, 0),
        scalar_range=(-2, 7),
    )
    return lambda: ours.elevation(**args), lambda: upstream.elevation(**args)


@case("compute_normals, 250k quads")
def normals_case():
    ours, upstream = grid(501, 501)
    return ours.compute_normals, upstream.compute_normals


@case("triangulate, 250k quads")
def triangulate_case():
    ours, upstream = grid(501, 501)
    return ours.triangulate, upstream.triangulate


@case("compute_cell_sizes, 250k quads")
def sizes_case():
    ours, upstream = grid(501, 501)
    return ours.compute_cell_sizes, upstream.compute_cell_sizes


@case("cell_data_to_point_data, 250k quads")
def cell_to_point_case():
    ours, upstream = grid(501, 501)
    values = np.ascontiguousarray(
        np.column_stack(
            (
                np.arange(ours.n_cells, dtype=float),
                np.ones(ours.n_cells),
                np.full(ours.n_cells, 2.0),
            )
        )
    )
    ours.cell_data["values"] = values
    upstream.cell_data["values"] = values
    return ours.cell_data_to_point_data, upstream.cell_data_to_point_data


@case("point_data_to_cell_data, 251k points")
def point_to_cell_case():
    ours, upstream = grid(501, 501)
    values = np.ascontiguousarray(
        np.column_stack(
            (
                np.arange(ours.n_points, dtype=float),
                np.ones(ours.n_points),
                np.full(ours.n_points, 2.0),
            )
        )
    )
    ours.point_data["values"] = values
    upstream.point_data["values"] = values
    return ours.point_data_to_cell_data, upstream.point_data_to_cell_data


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    print(
        f"Machine: {cpu_name()}; {os.cpu_count()} logical CPUs; "
        f"{platform.system()} {platform.machine()}; Python {platform.python_version()}"
    )
    print()
    print("| operation | Mojo | PyVista/VTK | PyVista / Mojo |")
    print("|---|---:|---:|---:|")
    for name, builder in CASES:
        ours, upstream = builder()
        ours()
        upstream()
        mojo_time = best_time(ours)
        pyvista_time = best_time(upstream)
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{pyvista_time * 1e3:.2f} ms | {pyvista_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
