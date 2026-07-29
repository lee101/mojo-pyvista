"""A focused polygonal `PolyData` implementation backed by Mojo kernels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy as _copy

import numpy as np

from ._lib import addr, f64, i64, lib


class DataSetAttributes(dict):
    """Dictionary-like numeric arrays associated with mesh points or cells."""

    def __setitem__(self, key, value):
        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError("mesh arrays must have at least one dimension")
        super().__setitem__(str(key), np.ascontiguousarray(array))


def _packed_faces(faces) -> np.ndarray:
    array = np.asarray(faces)
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise TypeError("faces must contain integers")
    if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
        raise OverflowError("face values must fit in int64")
    if array.ndim == 2:
        width = array.shape[1]
        packed = np.empty((len(array), width + 1), dtype=np.int64)
        packed[:, 0] = width
        packed[:, 1:] = array
        array = packed.ravel()
    array = i64(array).ravel()
    offset = 0
    while offset < len(array):
        count = int(array[offset])
        if count < 3 or offset + count >= len(array):
            raise ValueError("faces must be packed as [n, id0, ..., idN, n, ...]")
        offset += count + 1
    if offset != len(array):
        raise ValueError("invalid packed face array")
    return array


def _face_counts(faces: np.ndarray) -> np.ndarray:
    counts = []
    offset = 0
    while offset < len(faces):
        count = int(faces[offset])
        counts.append(count)
        offset += count + 1
    return np.asarray(counts, dtype=np.int64)


def _copy_arrays(source: Mapping[str, np.ndarray]) -> DataSetAttributes:
    result = DataSetAttributes()
    for name, array in source.items():
        result[name] = np.array(array, copy=True, order="C")
    return result


class PolyData:
    """Polygon-only counterpart of :class:`pyvista.PolyData`.

    The constructor accepts an ``(n, 3)`` point array and PyVista's packed
    face layout, or a rectangular ``(n_faces, vertices_per_face)`` array.
    """

    def __init__(
        self,
        var_inp=None,
        faces=None,
        deep: bool = False,
        force_float: bool = True,
    ):
        del force_float
        if isinstance(var_inp, PolyData):
            other = var_inp
            self.points = np.array(other.points, copy=deep, order="C")
            self.faces = np.array(other.faces, copy=deep, order="C")
            self._face_counts = np.array(other._face_counts, copy=deep)
            self._uniform_face_size = other._uniform_face_size
            self._point_offsets = (
                None
                if other._point_offsets is None
                else np.array(other._point_offsets, copy=deep)
            )
            self._incident_cells = (
                None
                if other._incident_cells is None
                else np.array(other._incident_cells, copy=deep)
            )
            if deep:
                self.point_data = _copy_arrays(other.point_data)
                self.cell_data = _copy_arrays(other.cell_data)
                self.field_data = _copy_arrays(other.field_data)
            else:
                self.point_data = DataSetAttributes(other.point_data)
                self.cell_data = DataSetAttributes(other.cell_data)
                self.field_data = DataSetAttributes(other.field_data)
            self._active_scalars_name = other._active_scalars_name
            self._active_scalars_association = other._active_scalars_association
            self._active_vectors_name = other._active_vectors_name
            return
        points = np.empty((0, 3), dtype=np.float64) if var_inp is None else f64(var_inp)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        self.points = np.array(points, copy=deep, order="C")
        self.faces = (
            np.empty(0, dtype=np.int64)
            if faces is None
            else np.array(_packed_faces(faces), copy=deep, order="C")
        )
        self._face_counts = _face_counts(self.faces)
        self._uniform_face_size = (
            int(self._face_counts[0])
            if self._face_counts.size
            and np.all(self._face_counts == self._face_counts[0])
            else 0
        )
        self._point_offsets: np.ndarray | None = None
        self._incident_cells: np.ndarray | None = None
        if self.faces.size and (
            self.faces[1:].min(initial=0) < 0
            or self.faces[1:].max(initial=-1) >= len(self.points)
        ):
            # Count entries can be larger than n_points, so validate ids precisely.
            offset = 0
            while offset < len(self.faces):
                count = int(self.faces[offset])
                ids = self.faces[offset + 1 : offset + count + 1]
                if np.any(ids < 0) or np.any(ids >= len(self.points)):
                    raise ValueError("face point index is out of range")
                offset += count + 1
        self.point_data = DataSetAttributes()
        self.cell_data = DataSetAttributes()
        self.field_data = DataSetAttributes()
        self._active_scalars_name: str | None = None
        self._active_scalars_association: str | None = None
        self._active_vectors_name: str | None = None

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return len(self._face_counts)

    @property
    def n_faces_strict(self) -> int:
        return self.n_cells

    @property
    def bounds(self) -> tuple[float, ...]:
        if not self.n_points:
            return (np.nan,) * 6
        return tuple(
            float(value)
            for value in (
                self.points[:, 0].min(),
                self.points[:, 0].max(),
                self.points[:, 1].min(),
                self.points[:, 1].max(),
                self.points[:, 2].min(),
                self.points[:, 2].max(),
            )
        )

    @property
    def center(self) -> tuple[float, float, float]:
        b = self.bounds
        return ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)

    @property
    def array_names(self) -> list[str]:
        return list(self.point_data) + list(self.cell_data) + list(self.field_data)

    @property
    def n_arrays(self) -> int:
        return len(self.array_names)

    @property
    def active_scalars_name(self) -> str | None:
        return self._active_scalars_name

    @property
    def active_scalars(self) -> np.ndarray | None:
        if self._active_scalars_name is None:
            return None
        data = (
            self.point_data
            if self._active_scalars_association == "point"
            else self.cell_data
        )
        return data.get(self._active_scalars_name)

    @property
    def active_vectors(self) -> np.ndarray | None:
        if self._active_vectors_name is None:
            return None
        return self.point_data.get(self._active_vectors_name)

    @property
    def point_normals(self) -> np.ndarray | None:
        return self.point_data.get("Normals")

    @property
    def cell_normals(self) -> np.ndarray | None:
        return self.cell_data.get("Normals")

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, name: str) -> np.ndarray:
        return self.get_array(name)

    def __setitem__(self, name: str, value):
        array = np.asarray(value)
        if array.ndim and len(array) == self.n_points:
            self.point_data[name] = array
            association = "point"
        elif array.ndim and len(array) == self.n_cells:
            self.cell_data[name] = array
            association = "cell"
        else:
            raise ValueError(
                f"array length must equal n_points ({self.n_points}) or "
                f"n_cells ({self.n_cells})"
            )
        if self._active_scalars_name is None:
            self.set_active_scalars(name, preference=association)

    def get_array(self, name: str, preference: str = "cell") -> np.ndarray:
        first, second = (
            (self.cell_data, self.point_data)
            if preference == "cell"
            else (self.point_data, self.cell_data)
        )
        if name in first:
            return first[name]
        if name in second:
            return second[name]
        if name in self.field_data:
            return self.field_data[name]
        raise KeyError(f"Data array ({name}) not present in this dataset.")

    def set_active_scalars(self, name: str, preference: str = "cell"):
        array = self.get_array(name, preference=preference)
        association = "point" if name in self.point_data and (
            preference == "point" or name not in self.cell_data
        ) else "cell"
        self._active_scalars_name = name
        self._active_scalars_association = association
        return array

    def set_active_vectors(self, name: str, preference: str = "point"):
        array = self.get_array(name, preference=preference)
        if array.ndim != 2 or array.shape[1] != 3 or len(array) != self.n_points:
            raise ValueError("active point vectors must have shape (n_points, 3)")
        self._active_vectors_name = name
        return array

    def copy(self, deep: bool = True) -> "PolyData":
        return PolyData(self, deep=deep)

    def _target(self, inplace: bool | None) -> "PolyData":
        return self if inplace is True else self.copy(deep=False)

    def _require_polygons(self):
        if not self.n_points or not self.n_cells:
            raise ValueError("this filter requires at least one polygonal face")

    def _validate_ffi_geometry(self, *, require_polygons: bool = False):
        """Validate every invariant assumed by kernels receiving raw addresses."""
        points = self.points
        faces = self.faces
        if (
            not isinstance(points, np.ndarray)
            or points.dtype != np.float64
            or points.ndim != 2
            or points.shape[1:] != (3,)
            or not points.flags.c_contiguous
        ):
            raise ValueError("points must be a C-contiguous float64 array with shape (n, 3)")
        if (
            not isinstance(faces, np.ndarray)
            or faces.dtype != np.int64
            or faces.ndim != 1
            or not faces.flags.c_contiguous
        ):
            raise ValueError("faces must be a C-contiguous one-dimensional int64 array")
        if self._uniform_face_size:
            width = self._uniform_face_size + 1
            if faces.size != self.n_cells * width:
                raise ValueError("faces were modified after mesh construction")
            rows = faces.reshape(self.n_cells, width)
            if np.any(rows[:, 0] != self._uniform_face_size):
                raise ValueError("faces were modified after mesh construction")
            ids = rows[:, 1:]
            if np.any(ids < 0) or np.any(ids >= len(points)):
                raise ValueError("face point index is out of range")
            validated_count = self.n_cells
        else:
            validated_counts = _face_counts(_packed_faces(faces))
            if not np.array_equal(validated_counts, self._face_counts):
                raise ValueError("faces were modified after mesh construction")
            offset = 0
            for count in validated_counts:
                ids = faces[offset + 1 : offset + int(count) + 1]
                if np.any(ids < 0) or np.any(ids >= len(points)):
                    raise ValueError("face point index is out of range")
                offset += int(count) + 1
            validated_count = len(validated_counts)
        if require_polygons and (not len(points) or not validated_count):
            raise ValueError("this filter requires at least one polygonal face")

    @staticmethod
    def _kernel_array(value, leading_size: int, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 0 or len(array) != leading_size:
            raise ValueError(f"{name} must contain one tuple per mesh element")
        if array.dtype.kind not in "biuf":
            raise TypeError(f"{name} must contain real numeric values")
        if array.dtype.kind == "f" and array.dtype.itemsize > 8:
            raise TypeError(f"{name} cannot be narrowed from {array.dtype} to float64")
        if array.dtype.kind in "iu" and array.size:
            limit = 2**53
            if np.any(array > limit) or (
                array.dtype.kind == "i" and np.any(array < -limit)
            ):
                raise OverflowError(f"{name} integers must be exactly representable as float64")
        return f64(array)

    def _geometry(self):
        self._validate_ffi_geometry(require_polygons=True)
        centers = np.empty((self.n_cells, 3), dtype=np.float64)
        normals = np.empty_like(centers)
        areas = np.empty(self.n_cells, dtype=np.float64)
        counts = np.empty(self.n_cells, dtype=np.float64)
        lib().mpv_cell_geometry(
            addr(self.points),
            addr(self.faces),
            self.n_cells,
            addr(centers),
            addr(normals),
            addr(areas),
            addr(counts),
        )
        return centers, normals, areas, counts

    def _ensure_point_adjacency(self):
        self._validate_ffi_geometry(require_polygons=True)
        self._point_offsets = np.empty(self.n_points + 1, dtype=np.int64)
        self._incident_cells = np.empty(
            len(self.faces) - self.n_cells, dtype=np.int64
        )
        lib().mpv_build_point_adjacency(
            addr(self.faces),
            addr(self._point_offsets),
            addr(self._incident_cells),
            self.n_points,
            self.n_cells,
        )

    def compute_normals(
        self,
        cell_normals: bool = True,
        point_normals: bool = True,
        split_vertices: bool = False,
        flip_normals: bool = False,
        consistent_normals: bool = True,
        auto_orient_normals: bool = False,
        non_manifold_traversal: bool = True,
        feature_angle=30.0,
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        del consistent_normals, non_manifold_traversal, feature_angle, progress_bar
        if split_vertices:
            raise NotImplementedError("split_vertices=True is not supported")
        if auto_orient_normals:
            raise NotImplementedError("auto_orient_normals=True is not supported")
        self._validate_ffi_geometry(require_polygons=True)
        result = self._target(inplace)
        _, cell_values, _, _ = result._geometry()
        point_values = np.empty((result.n_points, 3), dtype=np.float64)
        lib().mpv_point_normals(
            addr(result.faces),
            addr(cell_values),
            addr(point_values),
            result.n_points,
            result.n_cells,
            int(flip_normals),
        )
        if flip_normals:
            cell_values *= -1
        if cell_normals:
            result.cell_data["Normals"] = cell_values
        else:
            result.cell_data.pop("Normals", None)
        if point_normals:
            result.point_data["Normals"] = point_values
        else:
            result.point_data.pop("Normals", None)
        return result

    def triangulate(
        self,
        *,
        pass_verts: bool = False,
        pass_lines: bool = False,
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        del pass_verts, pass_lines, progress_bar
        self._validate_ffi_geometry(require_polygons=True)
        counts = self._face_counts
        triangles_per_cell = counts - 2
        packed = np.empty(int(triangles_per_cell.sum()) * 4, dtype=np.int64)
        written = lib().mpv_triangulate(
            addr(self.faces), self.n_cells, addr(packed)
        )
        result = self._target(inplace)
        result.faces = packed[:written]
        result._face_counts = np.full(int(triangles_per_cell.sum()), 3, dtype=np.int64)
        result._uniform_face_size = 3
        result._point_offsets = None
        result._incident_cells = None
        for name, array in list(result.cell_data.items()):
            result.cell_data[name] = np.repeat(array, triangles_per_cell, axis=0)
        return result

    def transform(
        self,
        trans,
        transform_all_input_vectors: bool = False,
        inplace: bool | None = None,
        progress_bar: bool = False,
    ):
        del progress_bar
        matrix = f64(getattr(trans, "matrix", trans))
        if matrix.shape != (4, 4):
            raise ValueError("trans must be a 4x4 transformation matrix")
        self._validate_ffi_geometry()
        result = self._target(inplace)
        transformed = np.empty_like(result.points)
        if result.n_points:
            lib().mpv_transform_points(
                addr(result.points), addr(transformed), result.n_points, addr(matrix)
            )
        result.points = transformed
        normal_matrix = np.eye(4)
        normal_matrix[:3, :3] = np.linalg.inv(matrix[:3, :3]).T
        for collection in (result.point_data, result.cell_data):
            for name, array in list(collection.items()):
                is_triplet = array.ndim == 2 and array.shape[1] == 3
                is_normal = is_triplet and name == "Normals"
                is_active_vector = (
                    collection is result.point_data
                    and name == result._active_vectors_name
                )
                if not is_triplet or not (
                    transform_all_input_vectors or is_normal or is_active_vector
                ):
                    continue
                src = self._kernel_array(array, len(array), name)
                dst = np.empty_like(src)
                applied_matrix = normal_matrix if is_normal else matrix
                lib().mpv_transform_vectors(
                    addr(src), addr(dst), len(src), addr(applied_matrix)
                )
                if is_normal:
                    lengths = np.linalg.norm(dst, axis=1)
                    nonzero = lengths != 0
                    dst[nonzero] /= lengths[nonzero, None]
                collection[name] = dst
        return result

    def warp_by_scalar(
        self,
        scalars: str | None = None,
        factor: float = 1.0,
        normal=None,
        inplace: bool = False,
        progress_bar: bool = False,
        **kwargs,
    ):
        del progress_bar, kwargs
        if scalars is None:
            if self.active_scalars is None:
                raise ValueError("no active scalars are set")
            values = self.active_scalars
            if self._active_scalars_association != "point":
                raise ValueError("warp_by_scalar requires point data")
        else:
            values = self.get_array(scalars, preference="point")
            if scalars not in self.point_data:
                raise ValueError("warp_by_scalar requires point data")
        self._validate_ffi_geometry()
        values = self._kernel_array(values, self.n_points, "scalars").reshape(-1)
        if len(values) != self.n_points:
            raise ValueError("scalars must contain one value per point")
        if normal is None:
            normals = self.point_normals
            if normals is None:
                normals = self.compute_normals().point_normals
        else:
            vector = f64(normal).reshape(-1)
            if vector.shape != (3,):
                raise ValueError("normal must contain three values")
            normals = np.tile(vector, (self.n_points, 1))
        normals = self._kernel_array(normals, self.n_points, "normals")
        if normals.shape != self.points.shape:
            raise ValueError("normals must have shape (n_points, 3)")
        result = self._target(inplace)
        points = np.empty_like(result.points)
        lib().mpv_warp_scalar(
            addr(result.points),
            addr(values),
            addr(normals),
            addr(points),
            result.n_points,
            factor,
        )
        result.points = points
        return result

    def warp_by_vector(
        self,
        vectors: str | None = None,
        factor: float = 1.0,
        inplace: bool = False,
        progress_bar: bool = False,
    ):
        del progress_bar
        if vectors is None:
            values = self.active_vectors
            if values is None:
                raise ValueError("no active point vectors are set")
        else:
            values = self.get_array(vectors, preference="point")
            if vectors not in self.point_data:
                raise ValueError("warp_by_vector requires point data")
        self._validate_ffi_geometry()
        values = self._kernel_array(values, self.n_points, "vectors")
        if values.shape != self.points.shape:
            raise ValueError("vectors must have shape (n_points, 3)")
        result = self._target(inplace)
        points = np.empty_like(result.points)
        lib().mpv_warp_vector(
            addr(result.points), addr(values), addr(points), result.n_points, factor
        )
        result.points = points
        return result

    def elevation(
        self,
        low_point=None,
        high_point=None,
        scalar_range=None,
        preference: str = "point",
        set_active: bool = True,
        progress_bar: bool = False,
    ):
        del progress_bar
        self._validate_ffi_geometry()
        if not self.n_points:
            return self.copy()
        bounds = (
            self.bounds
            if low_point is None or high_point is None or scalar_range is None
            else None
        )
        low = f64(
            (
                (bounds[0] + bounds[1]) * 0.5,
                (bounds[2] + bounds[3]) * 0.5,
                bounds[4],
            )
            if low_point is None
            else low_point
        ).reshape(-1)
        high = f64(
            (
                (bounds[0] + bounds[1]) * 0.5,
                (bounds[2] + bounds[3]) * 0.5,
                bounds[5],
            )
            if high_point is None
            else high_point
        ).reshape(-1)
        if low.shape != (3,) or high.shape != (3,):
            raise ValueError("low_point and high_point must contain three values")
        if scalar_range is None:
            range_values = (bounds[4], bounds[5])
        elif isinstance(scalar_range, str):
            source = self.get_array(scalar_range, preference=preference)
            range_values = (float(np.min(source)), float(np.max(source)))
        else:
            range_values = tuple(scalar_range)
        if len(range_values) != 2:
            raise ValueError("scalar_range must contain two values")
        result = self.copy(deep=False)
        values = np.empty(self.n_points, dtype=np.float64)
        lib().mpv_elevation(
            addr(self.points),
            addr(values),
            self.n_points,
            *low,
            *high,
            *range_values,
        )
        result.point_data["Elevation"] = values
        if set_active:
            result.set_active_scalars("Elevation", preference="point")
        return result

    def cell_data_to_point_data(
        self, pass_cell_data: bool = False, progress_bar: bool = False
    ):
        del progress_bar
        self._validate_ffi_geometry(require_polygons=True)
        self._ensure_point_adjacency()
        result = self.copy(deep=False)
        for name, source in self.cell_data.items():
            source_f64 = self._kernel_array(source, self.n_cells, name)
            flat = source_f64.reshape(self.n_cells, -1)
            converted = np.empty((self.n_points, flat.shape[1]), dtype=np.float64)
            lib().mpv_cell_to_point(
                addr(self._point_offsets),
                addr(self._incident_cells),
                addr(flat),
                addr(converted),
                self.n_points,
                flat.shape[1],
            )
            result.point_data[name] = (
                converted[:, 0] if source_f64.ndim == 1 else converted
            )
        if not pass_cell_data:
            result.cell_data.clear()
        if self._active_scalars_association == "cell" and self._active_scalars_name:
            result.set_active_scalars(self._active_scalars_name, preference="point")
        return result

    def point_data_to_cell_data(
        self,
        pass_point_data: bool = False,
        categorical: bool = False,
        progress_bar: bool = False,
    ):
        del progress_bar
        if categorical:
            raise NotImplementedError("categorical=True is not supported")
        self._validate_ffi_geometry(require_polygons=True)
        result = self.copy(deep=False)
        for name, source in self.point_data.items():
            source_f64 = self._kernel_array(source, self.n_points, name)
            flat = source_f64.reshape(self.n_points, -1)
            converted = np.empty((self.n_cells, flat.shape[1]), dtype=np.float64)
            lib().mpv_point_to_cell(
                addr(self.faces),
                addr(flat),
                addr(converted),
                self.n_cells,
                flat.shape[1],
                self._uniform_face_size,
            )
            result.cell_data[name] = (
                converted[:, 0] if source_f64.ndim == 1 else converted
            )
        if not pass_point_data:
            result.point_data.clear()
        if self._active_scalars_association == "point" and self._active_scalars_name:
            result.set_active_scalars(self._active_scalars_name, preference="cell")
        return result

    def compute_cell_sizes(
        self,
        length: bool = True,
        area: bool = True,
        volume: bool = True,
        progress_bar: bool = False,
        vertex_count: bool = False,
    ):
        del progress_bar
        self._validate_ffi_geometry()
        result = self.copy(deep=False)
        if length:
            result.cell_data["Length"] = np.zeros(self.n_cells)
        if area:
            areas = np.empty(self.n_cells, dtype=np.float64)
            if self.n_cells:
                lib().mpv_cell_areas(
                    addr(self.points),
                    addr(self.faces),
                    addr(areas),
                    self.n_cells,
                    self._uniform_face_size,
                )
            result.cell_data["Area"] = areas
        if volume:
            result.cell_data["Volume"] = np.zeros(self.n_cells)
        if vertex_count:
            result.cell_data["VertexCount"] = np.zeros(self.n_cells)
        return result

    def cell_centers(
        self,
        vertex: bool = True,
        pass_cell_data: bool = True,
        progress_bar: bool = False,
    ):
        del vertex, progress_bar
        centers, _, _, _ = self._geometry()
        result = PolyData(centers)
        if pass_cell_data:
            result.point_data = _copy_arrays(self.cell_data)
        return result


def wrap(dataset) -> PolyData:
    """Copy a polygonal PyVista-like object into :class:`PolyData`."""
    if isinstance(dataset, PolyData):
        return dataset
    unsupported = [
        name
        for name in ("n_verts", "n_lines", "n_strips")
        if int(getattr(dataset, name, 0)) != 0
    ]
    if unsupported:
        raise ValueError(
            "wrap only accepts polygon faces; unsupported topology is present: "
            + ", ".join(unsupported)
        )
    result = PolyData(dataset.points, dataset.faces)
    for name in dataset.point_data:
        result.point_data[name] = np.asarray(dataset.point_data[name])
    for name in dataset.cell_data:
        result.cell_data[name] = np.asarray(dataset.cell_data[name])
    active = getattr(dataset, "active_scalars_name", None)
    if active:
        preference = "point" if active in result.point_data else "cell"
        result.set_active_scalars(active, preference=preference)
    active_vectors = getattr(dataset, "active_vectors_name", None)
    if active_vectors and active_vectors in result.point_data:
        result.set_active_vectors(active_vectors)
    return result
