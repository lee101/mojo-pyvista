"""Numerical and behavioural parity with PyVista on polygonal meshes."""

import numpy as np
import pytest

import pyvista as pv
import mojopyvista as mpv


def pair(mesh):
    return mpv.wrap(mesh), mesh.copy()


def quad_grid(nx, ny):
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    points = np.ascontiguousarray(
        np.column_stack((x.ravel(), y.ravel(), np.zeros(nx * ny)))
    )
    rows = np.arange(ny - 1)[:, None]
    cols = np.arange(nx - 1)[None, :]
    p0 = (rows * nx + cols).ravel()
    cells = np.column_stack((p0, p0 + 1, p0 + nx + 1, p0 + nx))
    return mpv.PolyData(points, cells), cells


@pytest.fixture
def plane():
    return pair(pv.Plane(i_resolution=4, j_resolution=3))


@pytest.fixture
def sphere():
    mesh = pv.Sphere(theta_resolution=16, phi_resolution=10)
    mesh.point_data.pop("Normals")
    return pair(mesh)


def test_constructor_properties_and_packed_faces():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    )
    mesh = mpv.PolyData(points, [[0, 1, 2, 3]])
    assert mesh.n_points == 4
    assert mesh.n_cells == 1
    assert np.array_equal(mesh.faces, [4, 0, 1, 2, 3])
    assert mesh.bounds == (0, 1, 0, 1, 0, 0)
    assert mesh.center == (0.5, 0.5, 0.0)


def test_array_access_and_active_metadata(plane):
    ours, _ = plane
    ours["height"] = ours.points[:, 0]
    ours.point_data["velocity"] = np.ones((ours.n_points, 3))
    ours.set_active_scalars("height", preference="point")
    ours.set_active_vectors("velocity")
    assert ours.active_scalars is ours["height"]
    assert ours.active_vectors is ours["velocity"]
    assert ours.active_scalars_name == "height"
    assert ours.n_arrays == 4


def test_field_data_and_copy(plane):
    ours, _ = plane
    ours.field_data["metadata"] = np.array([1.0, 2.0])
    copied = ours.copy()
    assert np.array_equal(copied.field_data["metadata"], [1.0, 2.0])
    copied.field_data["metadata"][0] = -1
    assert ours.field_data["metadata"][0] == 1


@pytest.mark.parametrize("factory", [pv.Cube, lambda: pv.Plane(i_resolution=4)])
def test_compute_normals(factory):
    ours, upstream = pair(factory())
    ours.point_data.pop("Normals", None)
    upstream.point_data.pop("Normals", None)
    result = ours.compute_normals()
    expected = upstream.compute_normals(split_vertices=False)
    assert np.allclose(result.point_normals, expected.point_normals, atol=1e-7)
    assert np.allclose(result.cell_normals, expected.cell_normals, atol=1e-7)
    assert "Normals" not in ours.point_data


def test_compute_normals_curved_mesh(sphere):
    ours, upstream = sphere
    result = ours.compute_normals()
    expected = upstream.compute_normals(split_vertices=False)
    assert np.allclose(result.point_normals, expected.point_normals, atol=1e-7)
    assert np.allclose(result.cell_normals, expected.cell_normals, atol=1e-7)


def test_compute_normals_options_and_inplace(plane):
    ours, upstream = plane
    returned = ours.compute_normals(
        point_normals=False, flip_normals=True, inplace=True
    )
    expected = upstream.compute_normals(
        point_normals=False, flip_normals=True, inplace=False
    )
    assert returned is ours
    assert "Normals" not in ours.point_data
    assert np.allclose(ours.cell_normals, expected.cell_normals)


def test_triangulate_connectivity_and_cell_data(plane):
    ours, upstream = plane
    values = np.arange(ours.n_cells, dtype=float)
    ours.cell_data["value"] = values
    upstream.cell_data["value"] = values
    result = ours.triangulate()
    expected = upstream.triangulate()
    assert np.array_equal(result.faces, expected.faces)
    assert np.array_equal(result.cell_data["value"], expected.cell_data["value"])
    assert result.n_cells == 2 * ours.n_cells


def test_triangulate_triangles_unchanged():
    upstream = pv.Sphere(theta_resolution=8, phi_resolution=6)
    ours = mpv.wrap(upstream)
    assert np.array_equal(ours.triangulate().faces, upstream.triangulate().faces)


def test_convex_ngon_geometry_and_triangulation():
    angles = np.linspace(0, 2 * np.pi, 6)[:-1]
    points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(5)))
    faces = np.concatenate(([5], np.arange(5)))
    ours = mpv.PolyData(points, faces)
    upstream = pv.PolyData(points, faces)
    assert np.allclose(
        ours.compute_cell_sizes()["Area"],
        upstream.compute_cell_sizes()["Area"],
    )
    assert np.allclose(
        ours.triangulate().compute_cell_sizes()["Area"].sum(),
        upstream.compute_cell_sizes()["Area"].sum(),
    )


def test_transform_points(plane):
    ours, upstream = plane
    angle = np.deg2rad(31)
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 2],
            [np.sin(angle), np.cos(angle), 0, -3],
            [0, 0, 1.5, 4],
            [0, 0, 0, 1],
        ]
    )
    result = ours.transform(matrix, inplace=False)
    expected = upstream.transform(matrix, inplace=False)
    assert np.allclose(result.points, expected.points)
    assert np.array_equal(ours.points, mpv.wrap(pv.Plane(i_resolution=4, j_resolution=3)).points)


def test_transform_all_vectors(plane):
    ours, upstream = plane
    vectors = np.tile([1.0, 2.0, 3.0], (ours.n_points, 1))
    ours.point_data["v"] = vectors
    upstream.point_data["v"] = vectors
    matrix = np.diag([2.0, 3.0, 4.0, 1.0])
    result = ours.transform(
        matrix, transform_all_input_vectors=True, inplace=False
    )
    expected = upstream.transform(
        matrix, transform_all_input_vectors=True, inplace=False
    )
    assert np.allclose(result.points, expected.points)
    assert np.allclose(result.point_data["v"], expected.point_data["v"])
    assert np.allclose(result.point_normals, expected.point_normals)


def test_transform_active_vectors_without_all_flag(plane):
    ours, upstream = plane
    vectors = np.tile([1.0, 2.0, 3.0], (ours.n_points, 1))
    ours.point_data["v"] = vectors
    upstream.point_data["v"] = vectors
    ours.set_active_vectors("v")
    upstream.set_active_vectors("v")
    matrix = np.diag([2.0, 3.0, 4.0, 1.0])
    result = ours.transform(matrix, inplace=False)
    expected = upstream.transform(matrix, inplace=False)
    assert np.allclose(result.point_data["v"], expected.point_data["v"])
    assert np.allclose(result.point_normals, expected.point_normals)


def test_warp_by_scalar_using_normals(plane):
    ours, upstream = plane
    values = np.sin(ours.points[:, 0] * 3) + ours.points[:, 1]
    ours.point_data["s"] = values
    upstream.point_data["s"] = values
    ours.set_active_scalars("s", preference="point")
    upstream.set_active_scalars("s", preference="point")
    result = ours.warp_by_scalar(factor=2.5)
    expected = upstream.warp_by_scalar(factor=2.5)
    assert np.allclose(result.points, expected.points, atol=1e-7)


def test_warp_by_scalar_explicit_normal(plane):
    ours, upstream = plane
    values = np.arange(ours.n_points, dtype=float)
    ours.point_data["s"] = values
    upstream.point_data["s"] = values
    result = ours.warp_by_scalar("s", factor=0.25, normal=(1, 0, 0))
    expected = upstream.warp_by_scalar("s", factor=0.25, normal=(1, 0, 0))
    assert np.allclose(result.points, expected.points)


def test_warp_by_vector(plane):
    ours, upstream = plane
    vectors = np.column_stack(
        (ours.points[:, 1], -ours.points[:, 0], np.ones(ours.n_points))
    )
    ours.point_data["velocity"] = vectors
    upstream.point_data["velocity"] = vectors
    ours.set_active_vectors("velocity")
    upstream.set_active_vectors("velocity")
    result = ours.warp_by_vector(factor=0.75)
    expected = upstream.warp_by_vector(factor=0.75)
    assert np.allclose(result.points, expected.points)


def test_warp_by_vector_simd_tail():
    points = np.arange(21, dtype=float).reshape(7, 3)
    vectors = np.linspace(-2, 3, 21).reshape(7, 3)
    mesh = mpv.PolyData(points)
    mesh.point_data["v"] = vectors
    mesh.set_active_vectors("v")
    result = mesh.warp_by_vector(factor=0.125)
    assert np.allclose(result.points, points + 0.125 * vectors)


def test_warp_by_scalar_and_elevation_simd_tails():
    points = np.arange(21, dtype=float).reshape(7, 3)
    scalars = np.linspace(-2, 3, 7)
    mesh = mpv.PolyData(points)
    mesh.point_data["s"] = scalars
    warped = mesh.warp_by_scalar("s", factor=0.25, normal=(0, 1, 0))
    expected = points.copy()
    expected[:, 1] += 0.25 * scalars
    assert np.allclose(warped.points, expected)
    assert np.allclose(
        mesh.elevation(
            low_point=(0, 0, 0),
            high_point=(18, 0, 0),
            scalar_range=(-1, 2),
        )["Elevation"],
        -1 + np.clip(points[:, 0] / 18, 0, 1) * 3,
    )


def test_elevation_defaults():
    points = np.array([[0, 0, -2], [0, 0, 1], [0, 0, 6]], dtype=float)
    faces = [3, 0, 1, 2]
    ours = mpv.PolyData(points, faces)
    upstream = pv.PolyData(points, faces)
    assert np.allclose(ours.elevation()["Elevation"], upstream.elevation()["Elevation"])


def test_elevation_custom_range_and_direction(plane):
    ours, upstream = plane
    result = ours.elevation(
        low_point=(-0.5, 0, 0),
        high_point=(0.5, 0, 0),
        scalar_range=(-10, 20),
    )
    expected = upstream.elevation(
        low_point=(-0.5, 0, 0),
        high_point=(0.5, 0, 0),
        scalar_range=(-10, 20),
    )
    assert np.allclose(result["Elevation"], expected["Elevation"])
    assert result.active_scalars_name == expected.active_scalars_name


def test_elevation_range_from_named_array(plane):
    ours, upstream = plane
    values = np.linspace(7, 11, ours.n_cells)
    ours.cell_data["range_source"] = values
    upstream.cell_data["range_source"] = values
    result = ours.elevation(
        low_point=(-0.5, 0, 0),
        high_point=(0.5, 0, 0),
        scalar_range="range_source",
    )
    expected = upstream.elevation(
        low_point=(-0.5, 0, 0),
        high_point=(0.5, 0, 0),
        scalar_range="range_source",
    )
    assert np.allclose(result["Elevation"], expected["Elevation"])


def test_cell_data_to_point_data_scalar_and_vector(plane):
    ours, upstream = plane
    scalar = np.arange(ours.n_cells, dtype=float)
    vector = np.column_stack((scalar, scalar**2, -scalar))
    for mesh in (ours, upstream):
        mesh.cell_data["scalar"] = scalar
        mesh.cell_data["vector"] = vector
    result = ours.cell_data_to_point_data()
    expected = upstream.cell_data_to_point_data()
    assert np.allclose(result.point_data["scalar"], expected.point_data["scalar"])
    assert np.allclose(result.point_data["vector"], expected.point_data["vector"])
    assert not result.cell_data


def test_cell_data_to_point_data_passes_source(plane):
    ours, upstream = plane
    values = np.arange(ours.n_cells, dtype=float)
    ours.cell_data["values"] = values
    upstream.cell_data["values"] = values
    result = ours.cell_data_to_point_data(pass_cell_data=True)
    expected = upstream.cell_data_to_point_data(pass_cell_data=True)
    assert np.allclose(result["values"], expected["values"])
    assert np.array_equal(result.cell_data["values"], expected.cell_data["values"])


def test_point_data_to_cell_data_scalar_and_vector(plane):
    ours, upstream = plane
    scalar = np.arange(ours.n_points, dtype=float)
    vector = np.column_stack((scalar, scalar**2, -scalar))
    for mesh in (ours, upstream):
        mesh.point_data["scalar"] = scalar
        mesh.point_data["vector"] = vector
    result = ours.point_data_to_cell_data()
    expected = upstream.point_data_to_cell_data()
    assert np.allclose(result.cell_data["scalar"], expected.cell_data["scalar"])
    assert np.allclose(result.cell_data["vector"], expected.cell_data["vector"])
    assert not result.point_data


def test_point_data_to_cell_data_passes_source(plane):
    ours, upstream = plane
    values = np.arange(ours.n_points, dtype=float)
    ours.point_data["values"] = values
    upstream.point_data["values"] = values
    result = ours.point_data_to_cell_data(pass_point_data=True)
    expected = upstream.point_data_to_cell_data(pass_point_data=True)
    assert np.allclose(result.cell_data["values"], expected.cell_data["values"])
    assert np.array_equal(result.point_data["values"], expected.point_data["values"])


def test_compute_cell_sizes(plane):
    ours, upstream = plane
    result = ours.compute_cell_sizes(vertex_count=True)
    expected = upstream.compute_cell_sizes(vertex_count=True)
    for name in ("Length", "Area", "Volume", "VertexCount"):
        assert np.allclose(result.cell_data[name], expected.cell_data[name])


def test_mixed_faces_and_simd_component_tail():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    )
    faces = np.array([3, 0, 1, 2, 4, 0, 1, 2, 3])
    ours = mpv.PolyData(points, faces)
    upstream = pv.PolyData(points, faces)
    point_values = np.arange(20, dtype=float).reshape(4, 5)
    cell_values = np.arange(10, dtype=float).reshape(2, 5)
    for mesh in (ours, upstream):
        mesh.point_data["point_values"] = point_values
        mesh.cell_data["cell_values"] = cell_values
    assert np.allclose(
        ours.compute_cell_sizes()["Area"],
        upstream.compute_cell_sizes()["Area"],
    )
    assert np.allclose(
        ours.point_data_to_cell_data().cell_data["point_values"],
        upstream.point_data_to_cell_data().cell_data["point_values"],
    )
    assert np.allclose(
        ours.cell_data_to_point_data().point_data["cell_values"],
        upstream.cell_data_to_point_data().point_data["cell_values"],
    )


def test_parallel_threshold_paths_and_shallow_results():
    n_points = 100_003
    points = np.column_stack(
        (
            np.linspace(-1, 1, n_points),
            np.zeros(n_points),
            np.ones(n_points),
        )
    )
    mesh = mpv.PolyData(points)
    scalars = np.linspace(-2, 2, n_points)
    vectors = np.column_stack((scalars, -scalars, 0.5 * scalars))
    mesh.point_data["s"] = scalars
    mesh.point_data["v"] = vectors
    mesh.set_active_vectors("v")
    assert np.allclose(
        mesh.warp_by_vector(factor=0.25).points,
        points + 0.25 * vectors,
    )
    assert np.allclose(
        mesh.warp_by_scalar("s", factor=0.5, normal=(1, 0, 0)).points,
        points + np.column_stack((0.5 * scalars, scalars * 0, scalars * 0)),
    )
    expected_elevation = -3 + np.clip((points[:, 0] + 1) * 0.5, 0, 1) * 8
    assert np.allclose(
        mesh.elevation(
            low_point=(-1, 0, 0),
            high_point=(1, 0, 0),
            scalar_range=(-3, 5),
        )["Elevation"],
        expected_elevation,
    )

    grid_mesh, cells = quad_grid(318, 318)
    point_values = np.column_stack(
        tuple(np.arange(grid_mesh.n_points, dtype=float) + i for i in range(5))
    )
    cell_values = np.ones((grid_mesh.n_cells, 5))
    grid_mesh.point_data["point_values"] = point_values
    grid_mesh.cell_data["cell_values"] = cell_values
    point_result = grid_mesh.cell_data_to_point_data()
    cell_result = grid_mesh.point_data_to_cell_data()
    size_result = grid_mesh.compute_cell_sizes()
    assert np.allclose(point_result.point_data["cell_values"], 1)
    assert np.allclose(
        cell_result.cell_data["point_values"],
        point_values[cells].mean(axis=1),
    )
    assert np.allclose(size_result.cell_data["Area"], 1)
    for result in (point_result, cell_result, size_result):
        assert np.shares_memory(result.points, grid_mesh.points)
        assert np.shares_memory(result.faces, grid_mesh.faces)


def test_cell_centers_and_pass_data(plane):
    ours, upstream = plane
    values = np.arange(ours.n_cells, dtype=float)
    ours.cell_data["values"] = values
    upstream.cell_data["values"] = values
    result = ours.cell_centers(pass_cell_data=True)
    expected = upstream.cell_centers(pass_cell_data=True)
    assert np.allclose(result.points, expected.points)
    assert np.array_equal(result.point_data["values"], expected.point_data["values"])


def test_wrap_and_deep_copy_are_independent(plane):
    ours, _ = plane
    copied = ours.copy()
    copied.points[0] += 10
    assert not np.array_equal(copied.points, ours.points)


def test_unsupported_topology_options_are_explicit(plane):
    ours, _ = plane
    with pytest.raises(NotImplementedError, match="split_vertices"):
        ours.compute_normals(split_vertices=True)
    with pytest.raises(NotImplementedError, match="categorical"):
        ours.point_data_to_cell_data(categorical=True)


def test_ffi_rejects_replaced_or_mutated_geometry_buffers(plane):
    ours, _ = plane
    ours.points = ours.points.astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        ours.transform(np.eye(4))

    ours = mpv.wrap(pv.Plane(i_resolution=4, j_resolution=3))
    ours.faces = ours.faces.copy()
    ours.faces[1] = ours.n_points
    with pytest.raises(ValueError, match="out of range"):
        ours.compute_cell_sizes()


def test_ffi_rejects_malformed_kernel_arrays(plane):
    ours, _ = plane
    ours.point_data["Normals"] = np.ones((ours.n_points, 2))
    ours.point_data["s"] = np.ones(ours.n_points)
    with pytest.raises(ValueError, match="normals"):
        ours.warp_by_scalar("s")

    ours.cell_data["bad"] = np.ones(ours.n_cells - 1)
    with pytest.raises(ValueError, match="mesh element"):
        ours.cell_data_to_point_data()


def test_faces_reject_lossy_conversion():
    points = np.zeros((3, 3))
    with pytest.raises(TypeError, match="integers"):
        mpv.PolyData(points, [3.0, 0.0, 1.0, 2.0])
    with pytest.raises(OverflowError, match="int64"):
        mpv.PolyData(points, np.array([3, 0, 1, 2**63], dtype=np.uint64))
    with pytest.raises(TypeError, match="narrowed"):
        mpv.PolyData(points.astype(np.longdouble))


def test_wrap_rejects_topology_it_would_discard():
    line = pv.PolyData(np.array([[0.0, 0, 0], [1.0, 0, 0]]), lines=[2, 0, 1])
    with pytest.raises(ValueError, match="n_lines"):
        mpv.wrap(line)


def test_empty_mesh_filters_do_not_cross_ffi():
    mesh = mpv.PolyData()
    assert mesh.transform(np.eye(4)).points.shape == (0, 3)
    assert mesh.elevation().n_points == 0
    sizes = mesh.compute_cell_sizes(vertex_count=True)
    for name in ("Length", "Area", "Volume", "VertexCount"):
        assert sizes.cell_data[name].size == 0
