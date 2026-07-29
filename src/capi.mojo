"""Polygon-mesh kernels exported through a C ABI."""

from std.algorithm import parallelize
from std.ffi import external_call
from std.math import sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_POINT_THRESHOLD = 100_000
comptime PARALLEL_CELL_THRESHOLD = 32_768
comptime POINT_CHUNK_SIZE = 16_384
comptime CELL_CHUNK_SIZE = 4_096


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@export("mpv_parallel_init")
def parallel_init() abi("C") -> Int:
    return external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()


def normalize3(x: Float64, y: Float64, z: Float64, dst: FPtr, base: Int):
    var length = sqrt(x * x + y * y + z * z)
    if length == 0.0:
        dst[base] = 0.0
        dst[base + 1] = 0.0
        dst[base + 2] = 0.0
    else:
        dst[base] = x / length
        dst[base + 1] = y / length
        dst[base + 2] = z / length


@export("mpv_transform_points")
def transform_points(
    points_addr: Int, dst_addr: Int, n: Int, matrix_addr: Int
) abi("C"):
    var points = fp(points_addr)
    var dst = fp(dst_addr)
    var matrix = fp(matrix_addr)
    for i in range(n):
        var x = points[i * 3]
        var y = points[i * 3 + 1]
        var z = points[i * 3 + 2]
        var w = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15]
        var invw = 1.0 if w == 0.0 or w == 1.0 else 1.0 / w
        dst[i * 3] = (
            matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]
        ) * invw
        dst[i * 3 + 1] = (
            matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]
        ) * invw
        dst[i * 3 + 2] = (
            matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11]
        ) * invw


@export("mpv_transform_vectors")
def transform_vectors(
    vectors_addr: Int, dst_addr: Int, n: Int, matrix_addr: Int
) abi("C"):
    var vectors = fp(vectors_addr)
    var dst = fp(dst_addr)
    var matrix = fp(matrix_addr)
    for i in range(n):
        var x = vectors[i * 3]
        var y = vectors[i * 3 + 1]
        var z = vectors[i * 3 + 2]
        dst[i * 3] = matrix[0] * x + matrix[1] * y + matrix[2] * z
        dst[i * 3 + 1] = matrix[4] * x + matrix[5] * y + matrix[6] * z
        dst[i * 3 + 2] = matrix[8] * x + matrix[9] * y + matrix[10] * z


@export("mpv_warp_scalar")
def warp_scalar(
    points_addr: Int,
    scalars_addr: Int,
    normals_addr: Int,
    dst_addr: Int,
    n: Int,
    factor: Float64,
) abi("C"):
    var points = fp(points_addr)
    var scalars = fp(scalars_addr)
    var normals = fp(normals_addr)
    var dst = fp(dst_addr)

    @parameter
    def work(chunk: Int):
        comptime W = simdwidthof[DType.float64]()
        var work_points = fp(points_addr)
        var work_scalars = fp(scalars_addr)
        var work_normals = fp(normals_addr)
        var work_dst = fp(dst_addr)
        var start = chunk * POINT_CHUNK_SIZE
        var end = min(start + POINT_CHUNK_SIZE, n)
        var vector_end = start + (end - start) // W * W
        for i in range(start, vector_end, W):
            var amounts = factor * work_scalars.load[width=W](i)
            for coord in range(3):
                var base = i * 3 + coord
                (work_dst + base).strided_store[width=W](
                    (work_points + base).strided_load[width=W](3)
                    + amounts * (work_normals + base).strided_load[width=W](3),
                    3,
                )
        for i in range(vector_end, end):
            var amount = factor * work_scalars[i]
            work_dst[i * 3] = work_points[i * 3] + amount * work_normals[i * 3]
            work_dst[i * 3 + 1] = (
                work_points[i * 3 + 1] + amount * work_normals[i * 3 + 1]
            )
            work_dst[i * 3 + 2] = (
                work_points[i * 3 + 2] + amount * work_normals[i * 3 + 2]
            )

    if n >= PARALLEL_POINT_THRESHOLD:
        parallelize[work]((n + POINT_CHUNK_SIZE - 1) // POINT_CHUNK_SIZE)
    else:
        comptime W = simdwidthof[DType.float64]()
        var vector_end = n // W * W
        for i in range(0, vector_end, W):
            var amounts = factor * scalars.load[width=W](i)
            for coord in range(3):
                var base = i * 3 + coord
                (dst + base).strided_store[width=W](
                    (points + base).strided_load[width=W](3)
                    + amounts * (normals + base).strided_load[width=W](3),
                    3,
                )
        for i in range(vector_end, n):
            var amount = factor * scalars[i]
            dst[i * 3] = points[i * 3] + amount * normals[i * 3]
            dst[i * 3 + 1] = points[i * 3 + 1] + amount * normals[i * 3 + 1]
            dst[i * 3 + 2] = points[i * 3 + 2] + amount * normals[i * 3 + 2]


@export("mpv_warp_vector")
def warp_vector(
    points_addr: Int,
    vectors_addr: Int,
    dst_addr: Int,
    n: Int,
    factor: Float64,
) abi("C"):
    var points = fp(points_addr)
    var vectors = fp(vectors_addr)
    var dst = fp(dst_addr)
    var size = n * 3

    @parameter
    def work(chunk: Int):
        comptime W = simdwidthof[DType.float64]()
        var work_points = fp(points_addr)
        var work_vectors = fp(vectors_addr)
        var work_dst = fp(dst_addr)
        var start = chunk * POINT_CHUNK_SIZE * 3
        var end = min(start + POINT_CHUNK_SIZE * 3, size)
        var vector_end = start + (end - start) // W * W
        for i in range(start, vector_end, W):
            work_dst.store(
                i,
                work_points.load[width=W](i)
                + factor * work_vectors.load[width=W](i),
            )
        for i in range(vector_end, end):
            work_dst[i] = work_points[i] + factor * work_vectors[i]

    if n >= PARALLEL_POINT_THRESHOLD:
        parallelize[work]((n + POINT_CHUNK_SIZE - 1) // POINT_CHUNK_SIZE)
    else:
        comptime W = simdwidthof[DType.float64]()
        var vector_end = size // W * W
        for i in range(0, vector_end, W):
            dst.store(
                i,
                points.load[width=W](i) + factor * vectors.load[width=W](i),
            )
        for i in range(vector_end, size):
            dst[i] = points[i] + factor * vectors[i]


@export("mpv_elevation")
def elevation(
    points_addr: Int,
    dst_addr: Int,
    n: Int,
    low_x: Float64,
    low_y: Float64,
    low_z: Float64,
    high_x: Float64,
    high_y: Float64,
    high_z: Float64,
    range_low: Float64,
    range_high: Float64,
) abi("C"):
    var points = fp(points_addr)
    var dst = fp(dst_addr)
    var dx = high_x - low_x
    var dy = high_y - low_y
    var dz = high_z - low_z
    var denom = dx * dx + dy * dy + dz * dz
    var output_scale = range_high - range_low

    @parameter
    def work(chunk: Int):
        var work_points = fp(points_addr)
        var work_dst = fp(dst_addr)
        var start = chunk * POINT_CHUNK_SIZE
        var end = min(start + POINT_CHUNK_SIZE, n)
        if denom == 0.0:
            comptime W = simdwidthof[DType.float64]()
            var vector_end = start + (end - start) // W * W
            for i in range(start, vector_end, W):
                work_dst.store(i, SIMD[DType.float64, W](range_low))
            for i in range(vector_end, end):
                work_dst[i] = range_low
            return
        comptime W = simdwidthof[DType.float64]()
        var inv_denom = 1.0 / denom
        var vector_end = start + (end - start) // W * W
        for i in range(start, vector_end, W):
            var t = (
                ((work_points + i * 3).strided_load[width=W](3) - low_x) * dx
                + ((work_points + i * 3 + 1).strided_load[width=W](3) - low_y)
                * dy
                + ((work_points + i * 3 + 2).strided_load[width=W](3) - low_z)
                * dz
            ) * inv_denom
            t = min(
                max(t, SIMD[DType.float64, W](0.0)),
                SIMD[DType.float64, W](1.0),
            )
            work_dst.store(i, range_low + t * output_scale)
        for i in range(vector_end, end):
            var t = (
                (work_points[i * 3] - low_x) * dx
                + (work_points[i * 3 + 1] - low_y) * dy
                + (work_points[i * 3 + 2] - low_z) * dz
            ) * inv_denom
            t = min(max(t, 0.0), 1.0)
            work_dst[i] = range_low + t * output_scale

    if n >= PARALLEL_POINT_THRESHOLD:
        parallelize[work]((n + POINT_CHUNK_SIZE - 1) // POINT_CHUNK_SIZE)
    else:
        if denom == 0.0:
            comptime W = simdwidthof[DType.float64]()
            var vector_end = n // W * W
            for i in range(0, vector_end, W):
                dst.store(i, SIMD[DType.float64, W](range_low))
            for i in range(vector_end, n):
                dst[i] = range_low
            return
        comptime W = simdwidthof[DType.float64]()
        var inv_denom = 1.0 / denom
        var vector_end = n // W * W
        for i in range(0, vector_end, W):
            var t = (
                ((points + i * 3).strided_load[width=W](3) - low_x) * dx
                + ((points + i * 3 + 1).strided_load[width=W](3) - low_y) * dy
                + ((points + i * 3 + 2).strided_load[width=W](3) - low_z) * dz
            ) * inv_denom
            t = min(
                max(t, SIMD[DType.float64, W](0.0)),
                SIMD[DType.float64, W](1.0),
            )
            dst.store(i, range_low + t * output_scale)
        for i in range(vector_end, n):
            var t = (
                (points[i * 3] - low_x) * dx
                + (points[i * 3 + 1] - low_y) * dy
                + (points[i * 3 + 2] - low_z) * dz
            ) * inv_denom
            t = min(max(t, 0.0), 1.0)
            dst[i] = range_low + t * output_scale


def cell_areas_range(
    points: FPtr,
    faces: IPtr,
    areas: FPtr,
    start: Int,
    end: Int,
    count: Int,
):
    for cell in range(start, end):
        var offset = cell * (count + 1)
        var last_idx = Int(faces[offset + count])
        var ax = points[last_idx * 3]
        var ay = points[last_idx * 3 + 1]
        var az = points[last_idx * 3 + 2]
        var nx = 0.0
        var ny = 0.0
        var nz = 0.0
        for j in range(count):
            var idx = Int(faces[offset + 1 + j])
            var bx = points[idx * 3]
            var by = points[idx * 3 + 1]
            var bz = points[idx * 3 + 2]
            nx += (ay - by) * (az + bz)
            ny += (az - bz) * (ax + bx)
            nz += (ax - bx) * (ay + by)
            ax = bx
            ay = by
            az = bz
        areas[cell] = 0.5 * sqrt(nx * nx + ny * ny + nz * nz)


@export("mpv_cell_areas")
def cell_areas(
    points_addr: Int,
    faces_addr: Int,
    areas_addr: Int,
    ncells: Int,
    uniform_count: Int,
) abi("C"):
    var points = fp(points_addr)
    var faces = ip(faces_addr)
    var areas = fp(areas_addr)
    if uniform_count > 0:

        @parameter
        def work(chunk: Int):
            var work_points = fp(points_addr)
            var work_faces = ip(faces_addr)
            var work_areas = fp(areas_addr)
            var start = chunk * CELL_CHUNK_SIZE
            var end = min(start + CELL_CHUNK_SIZE, ncells)
            cell_areas_range(
                work_points,
                work_faces,
                work_areas,
                start,
                end,
                uniform_count,
            )

        if ncells >= PARALLEL_CELL_THRESHOLD:
            parallelize[work]((ncells + CELL_CHUNK_SIZE - 1) // CELL_CHUNK_SIZE)
        else:
            cell_areas_range(points, faces, areas, 0, ncells, uniform_count)
        return

    var offset = 0
    for cell in range(ncells):
        var count = Int(faces[offset])
        var last_idx = Int(faces[offset + count])
        var ax = points[last_idx * 3]
        var ay = points[last_idx * 3 + 1]
        var az = points[last_idx * 3 + 2]
        var nx = 0.0
        var ny = 0.0
        var nz = 0.0
        for j in range(count):
            var idx = Int(faces[offset + 1 + j])
            var bx = points[idx * 3]
            var by = points[idx * 3 + 1]
            var bz = points[idx * 3 + 2]
            nx += (ay - by) * (az + bz)
            ny += (az - bz) * (ax + bx)
            nz += (ax - bx) * (ay + by)
            ax = bx
            ay = by
            az = bz
        areas[cell] = 0.5 * sqrt(nx * nx + ny * ny + nz * nz)
        offset += count + 1


@export("mpv_cell_geometry")
def cell_geometry(
    points_addr: Int,
    faces_addr: Int,
    ncells: Int,
    centers_addr: Int,
    normals_addr: Int,
    areas_addr: Int,
    vertex_counts_addr: Int,
) abi("C"):
    var points = fp(points_addr)
    var faces = ip(faces_addr)
    var centers = fp(centers_addr)
    var normals = fp(normals_addr)
    var areas = fp(areas_addr)
    var vertex_counts = fp(vertex_counts_addr)
    var offset = 0
    for cell in range(ncells):
        var count = Int(faces[offset])
        vertex_counts[cell] = Float64(count)
        var cx = 0.0
        var cy = 0.0
        var cz = 0.0
        for j in range(count):
            var idx = Int(faces[offset + 1 + j])
            cx += points[idx * 3]
            cy += points[idx * 3 + 1]
            cz += points[idx * 3 + 2]
        if count > 0:
            cx /= Float64(count)
            cy /= Float64(count)
            cz /= Float64(count)
        centers[cell * 3] = cx
        centers[cell * 3 + 1] = cy
        centers[cell * 3 + 2] = cz

        var nx = 0.0
        var ny = 0.0
        var nz = 0.0
        for j in range(count):
            var ia = Int(faces[offset + 1 + j])
            var ib = Int(faces[offset + 1 + (j + 1) % count])
            var ax = points[ia * 3]
            var ay = points[ia * 3 + 1]
            var az = points[ia * 3 + 2]
            var bx = points[ib * 3]
            var by = points[ib * 3 + 1]
            var bz = points[ib * 3 + 2]
            nx += (ay - by) * (az + bz)
            ny += (az - bz) * (ax + bx)
            nz += (ax - bx) * (ay + by)
        var doubled_area = sqrt(nx * nx + ny * ny + nz * nz)
        areas[cell] = 0.5 * doubled_area
        normalize3(nx, ny, nz, normals, cell * 3)
        offset += count + 1


@export("mpv_point_normals")
def point_normals(
    faces_addr: Int,
    cell_normals_addr: Int,
    point_normals_addr: Int,
    npoints: Int,
    ncells: Int,
    flip: Int,
) abi("C"):
    var faces = ip(faces_addr)
    var cell_normals = fp(cell_normals_addr)
    var point_normals = fp(point_normals_addr)
    for i in range(npoints * 3):
        point_normals[i] = 0.0
    var sign = -1.0 if flip != 0 else 1.0
    var offset = 0
    for cell in range(ncells):
        var count = Int(faces[offset])
        for j in range(count):
            var idx = Int(faces[offset + 1 + j])
            point_normals[idx * 3] += sign * cell_normals[cell * 3]
            point_normals[idx * 3 + 1] += sign * cell_normals[cell * 3 + 1]
            point_normals[idx * 3 + 2] += sign * cell_normals[cell * 3 + 2]
        offset += count + 1
    for i in range(npoints):
        normalize3(
            point_normals[i * 3],
            point_normals[i * 3 + 1],
            point_normals[i * 3 + 2],
            point_normals,
            i * 3,
        )


@export("mpv_triangulate")
def triangulate(faces_addr: Int, ncells: Int, dst_addr: Int) abi("C") -> Int:
    var faces = ip(faces_addr)
    var dst = ip(dst_addr)
    var src_offset = 0
    var dst_offset = 0
    for cell in range(ncells):
        var count = Int(faces[src_offset])
        if count == 4:
            dst[dst_offset] = 3
            dst[dst_offset + 1] = faces[src_offset + 1]
            dst[dst_offset + 2] = faces[src_offset + 2]
            dst[dst_offset + 3] = faces[src_offset + 4]
            dst[dst_offset + 4] = 3
            dst[dst_offset + 5] = faces[src_offset + 2]
            dst[dst_offset + 6] = faces[src_offset + 3]
            dst[dst_offset + 7] = faces[src_offset + 4]
            dst_offset += 8
        else:
            for j in range(1, count - 1):
                dst[dst_offset] = 3
                dst[dst_offset + 1] = faces[src_offset + 1]
                dst[dst_offset + 2] = faces[src_offset + 1 + j]
                dst[dst_offset + 3] = faces[src_offset + 2 + j]
                dst_offset += 4
        src_offset += count + 1
    return dst_offset


@export("mpv_cell_to_point")
def cell_to_point(
    point_offsets_addr: Int,
    incident_cells_addr: Int,
    src_addr: Int,
    dst_addr: Int,
    npoints: Int,
    ncomp: Int,
) abi("C"):
    var point_offsets = ip(point_offsets_addr)
    var incident_cells = ip(incident_cells_addr)
    var src = fp(src_addr)
    var dst = fp(dst_addr)

    @parameter
    def work(chunk: Int):
        comptime W = simdwidthof[DType.float64]()
        var work_offsets = ip(point_offsets_addr)
        var work_cells = ip(incident_cells_addr)
        var work_src = fp(src_addr)
        var work_dst = fp(dst_addr)
        var start = chunk * POINT_CHUNK_SIZE
        var end = min(start + POINT_CHUNK_SIZE, npoints)
        for point in range(start, end):
            var begin = Int(work_offsets[point])
            var incident_end = Int(work_offsets[point + 1])
            var count = incident_end - begin
            var comp = 0
            if count > 0:
                for vector_comp in range(0, ncomp // W * W, W):
                    var cell = Int(work_cells[begin])
                    var acc = work_src.load[width=W](cell * ncomp + vector_comp)
                    for incident in range(begin + 1, incident_end):
                        cell = Int(work_cells[incident])
                        acc += work_src.load[width=W](
                            cell * ncomp + vector_comp
                        )
                    work_dst.store(
                        point * ncomp + vector_comp,
                        acc / Float64(count),
                    )
                    comp += W
            for scalar_comp in range(comp, ncomp):
                var acc = 0.0
                for incident in range(begin, incident_end):
                    var cell = Int(work_cells[incident])
                    acc += work_src[cell * ncomp + scalar_comp]
                work_dst[point * ncomp + scalar_comp] = (
                    acc / Float64(count) if count > 0 else 0.0
                )

    if npoints >= PARALLEL_POINT_THRESHOLD:
        parallelize[work]((npoints + POINT_CHUNK_SIZE - 1) // POINT_CHUNK_SIZE)
    else:
        comptime W = simdwidthof[DType.float64]()
        for point in range(npoints):
            var begin = Int(point_offsets[point])
            var incident_end = Int(point_offsets[point + 1])
            var count = incident_end - begin
            var comp = 0
            if count > 0:
                for vector_comp in range(0, ncomp // W * W, W):
                    var cell = Int(incident_cells[begin])
                    var acc = src.load[width=W](cell * ncomp + vector_comp)
                    for incident in range(begin + 1, incident_end):
                        cell = Int(incident_cells[incident])
                        acc += src.load[width=W](cell * ncomp + vector_comp)
                    dst.store(
                        point * ncomp + vector_comp,
                        acc / Float64(count),
                    )
                    comp += W
            for scalar_comp in range(comp, ncomp):
                var acc = 0.0
                for incident in range(begin, incident_end):
                    var cell = Int(incident_cells[incident])
                    acc += src[cell * ncomp + scalar_comp]
                dst[point * ncomp + scalar_comp] = (
                    acc / Float64(count) if count > 0 else 0.0
                )


@export("mpv_build_point_adjacency")
def build_point_adjacency(
    faces_addr: Int,
    point_offsets_addr: Int,
    incident_cells_addr: Int,
    npoints: Int,
    ncells: Int,
) abi("C"):
    var faces = ip(faces_addr)
    var point_offsets = ip(point_offsets_addr)
    var incident_cells = ip(incident_cells_addr)
    for point in range(npoints + 1):
        point_offsets[point] = 0
    var face_offset = 0
    for cell in range(ncells):
        var count = Int(faces[face_offset])
        for j in range(count):
            var point = Int(faces[face_offset + 1 + j])
            point_offsets[point + 1] += 1
        face_offset += count + 1
    for point in range(npoints):
        point_offsets[point + 1] += point_offsets[point]
    face_offset = 0
    for cell in range(ncells):
        var count = Int(faces[face_offset])
        for j in range(count):
            var point = Int(faces[face_offset + 1 + j])
            var position = Int(point_offsets[point])
            incident_cells[position] = Int64(cell)
            point_offsets[point] += 1
        face_offset += count + 1
    for point in range(npoints, 0, -1):
        point_offsets[point] = point_offsets[point - 1]
    point_offsets[0] = 0


@export("mpv_point_to_cell")
def point_to_cell(
    faces_addr: Int,
    src_addr: Int,
    dst_addr: Int,
    ncells: Int,
    ncomp: Int,
    uniform_count: Int,
) abi("C"):
    var faces = ip(faces_addr)
    var src = fp(src_addr)
    var dst = fp(dst_addr)

    @parameter
    def uniform_work(chunk: Int):
        comptime W = simdwidthof[DType.float64]()
        var work_faces = ip(faces_addr)
        var work_src = fp(src_addr)
        var work_dst = fp(dst_addr)
        var start = chunk * CELL_CHUNK_SIZE
        var end = min(start + CELL_CHUNK_SIZE, ncells)
        for cell in range(start, end):
            var offset = cell * (uniform_count + 1)
            var comp = 0
            for vector_comp in range(0, ncomp // W * W, W):
                var first_idx = Int(work_faces[offset + 1])
                var acc = work_src.load[width=W](
                    first_idx * ncomp + vector_comp
                )
                for j in range(1, uniform_count):
                    var idx = Int(work_faces[offset + 1 + j])
                    acc += work_src.load[width=W](idx * ncomp + vector_comp)
                work_dst.store(
                    cell * ncomp + vector_comp,
                    acc / Float64(uniform_count),
                )
                comp += W
            for scalar_comp in range(comp, ncomp):
                var acc = 0.0
                for j in range(uniform_count):
                    var idx = Int(work_faces[offset + 1 + j])
                    acc += work_src[idx * ncomp + scalar_comp]
                work_dst[cell * ncomp + scalar_comp] = acc / Float64(
                    uniform_count
                )

    if uniform_count > 0:
        if ncells >= PARALLEL_CELL_THRESHOLD:
            parallelize[uniform_work](
                (ncells + CELL_CHUNK_SIZE - 1) // CELL_CHUNK_SIZE
            )
        else:
            comptime W = simdwidthof[DType.float64]()
            for cell in range(ncells):
                var offset = cell * (uniform_count + 1)
                var comp = 0
                for vector_comp in range(0, ncomp // W * W, W):
                    var first_idx = Int(faces[offset + 1])
                    var acc = src.load[width=W](first_idx * ncomp + vector_comp)
                    for j in range(1, uniform_count):
                        var idx = Int(faces[offset + 1 + j])
                        acc += src.load[width=W](idx * ncomp + vector_comp)
                    dst.store(
                        cell * ncomp + vector_comp,
                        acc / Float64(uniform_count),
                    )
                    comp += W
                for scalar_comp in range(comp, ncomp):
                    var acc = 0.0
                    for j in range(uniform_count):
                        var idx = Int(faces[offset + 1 + j])
                        acc += src[idx * ncomp + scalar_comp]
                    dst[cell * ncomp + scalar_comp] = acc / Float64(
                        uniform_count
                    )
        return

    var offset = 0
    for cell in range(ncells):
        var count = Int(faces[offset])
        comptime W = simdwidthof[DType.float64]()
        var comp = 0
        if count > 0:
            for vector_comp in range(0, ncomp // W * W, W):
                var first_idx = Int(faces[offset + 1])
                var acc = src.load[width=W](first_idx * ncomp + vector_comp)
                for j in range(1, count):
                    var idx = Int(faces[offset + 1 + j])
                    acc += src.load[width=W](idx * ncomp + vector_comp)
                dst.store(
                    cell * ncomp + vector_comp,
                    acc / Float64(count),
                )
                comp += W
        for scalar_comp in range(comp, ncomp):
            var acc = 0.0
            for j in range(count):
                var idx = Int(faces[offset + 1 + j])
                acc += src[idx * ncomp + scalar_comp]
            dst[cell * ncomp + scalar_comp] = (
                acc / Float64(count) if count > 0 else 0.0
            )
        offset += count + 1
