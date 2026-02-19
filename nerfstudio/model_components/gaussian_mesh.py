"""Gaussian mesh extraction via volumetric density evaluation + marching cubes.

Pipeline
--------
1. Compute an axis-aligned bounding box (AABB) from the Gaussian means (+ padding).
2. Build a uniform 3-D grid (e.g. 128³) inside the AABB.
3. Evaluate the volumetric density field  V(x) = Σ_i α_i  exp(−½ ‖diag(1/s_i) R_i^T (x − μ_i)‖²)
   for every grid point, with per-chunk AABB culling so only nearby
   Gaussians are considered.
4. Run *marching cubes* (scikit-image) at a user-chosen iso-value τ.
5. Export the resulting triangle mesh as PLY via Open3D.

Both ``scikit-image`` (for marching cubes) and ``open3d`` (for PLY I/O)
are required.  If either is missing the functions log an error and return
``None`` / ``[]``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert (w, x, y, z) quaternions to rotation matrices.

    Args:
        quats: (N, 4) quaternions (need not be unit — normalised internally).

    Returns:
        (N, 3, 3) rotation matrices.
    """
    q = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


@torch.no_grad()
def evaluate_gaussian_field(
    grid_points: torch.Tensor,
    means: torch.Tensor,
    rotmats: torch.Tensor,
    inv_scales: torch.Tensor,
    alphas: torch.Tensor,
    gauss_aabb_min: torch.Tensor,
    gauss_aabb_max: torch.Tensor,
    chunk_size: int = 16**3,
) -> torch.Tensor:
    """Evaluate Gaussian density V(x) on a flat array of grid points.

    For every chunk of grid points the function first culls Gaussians whose
    axis-aligned bounding box does not overlap the chunk's AABB, then evaluates
    only the surviving Gaussians.  When the number of surviving Gaussians is
    still large the evaluation is further sub-chunked over Gaussians to bound
    peak GPU memory.

    Args:
        grid_points: (P, 3) query positions.
        means:       (N, 3) Gaussian centres.
        rotmats:     (N, 3, 3) rotation matrices.
        inv_scales:  (N, 3) = 1 / exp(scales).
        alphas:      (N,) sigmoid opacities.
        gauss_aabb_min / gauss_aabb_max: (N, 3) per-Gaussian culling AABBs.
        chunk_size:  grid points per evaluation chunk.

    Returns:
        (P,) density values.
    """
    P = grid_points.shape[0]
    device = grid_points.device
    dtype = grid_points.dtype
    volume = torch.zeros(P, device=device, dtype=dtype)

    max_gauss_per_sub = max(1, (2**23) // max(1, chunk_size))

    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        pts = grid_points[start:end]  # (C, 3)
        C = pts.shape[0]

        c_min = pts.min(dim=0).values
        c_max = pts.max(dim=0).values

        mask = (gauss_aabb_max >= c_min).all(dim=1) & (gauss_aabb_min <= c_max).all(dim=1)
        if not mask.any():
            continue

        m = means[mask]
        r = rotmats[mask]
        is_ = inv_scales[mask]
        a = alphas[mask]
        M = int(m.shape[0])

        contrib = torch.zeros(C, device=device, dtype=dtype)
        for gs in range(0, M, max_gauss_per_sub):
            ge = min(gs + max_gauss_per_sub, M)
            d = pts[:, None, :] - m[None, gs:ge, :]  # (C, Msub, 3)
            d = torch.einsum("cmk,mkj->cmj", d, r[gs:ge])
            d = d * is_[None, gs:ge, :]
            maha = (d * d).sum(dim=-1)  # (C, Msub)
            g = torch.exp(-0.5 * maha)
            contrib += (a[None, gs:ge] * g).sum(dim=1)

        volume[start:end] = contrib

    return volume


@torch.no_grad()
def extract_mesh_from_gaussians(
    means: torch.Tensor,
    scales_exp: torch.Tensor,
    quats: torch.Tensor,
    opacities_sigmoid: torch.Tensor,
    resolution: int = 128,
    isovalue: float = 1.0,
    padding_factor: float = 0.05,
    culling_sigma: float = 3.0,
    chunk_size: int = 16**3,
) -> Optional[Tuple]:
    """Full pipeline: AABB → 3-D grid → density evaluation → marching cubes.

    Args:
        means:              (N, 3) Gaussian centres.
        scales_exp:         (N, 3) world-space scales (already ``exp``'d).
        quats:              (N, 4) quaternions (w, x, y, z).
        opacities_sigmoid:  (N,) or (N, 1) opacities in [0, 1].
        resolution:         voxel grid resolution per axis.
        isovalue:           marching-cubes iso-level τ.
        padding_factor:     fractional AABB padding.
        culling_sigma:      Gaussian culling radius in multiples of σ.
        chunk_size:         grid points per evaluation chunk.

    Returns:
        ``(vertices, faces)`` as numpy arrays, or *None* on failure.
    """
    try:
        from skimage.measure import marching_cubes  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "scikit-image is required for marching cubes. "
            "Install with: pip install scikit-image"
        )
        return None

    import numpy as np

    device = means.device
    alpha = opacities_sigmoid.view(-1)

    # ── AABB ──
    mins = means.min(dim=0).values
    maxs = means.max(dim=0).values
    pad = padding_factor * (maxs - mins).max()
    mins = mins - pad
    maxs = maxs + pad

    # ── Uniform grid ──
    lin = [
        torch.linspace(float(mins[i]), float(maxs[i]), resolution, device=device)
        for i in range(3)
    ]
    gx, gy, gz = torch.meshgrid(*lin, indexing="ij")
    grid_flat = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    voxel_size = ((maxs - mins) / max(resolution - 1, 1)).cpu().numpy()

    # ── Pre-compute Gaussian helpers ──
    R = _quat_to_rotmat(quats)
    inv_s = 1.0 / scales_exp.clamp(min=1e-8)
    g_min = means - culling_sigma * scales_exp
    g_max = means + culling_sigma * scales_exp

    logger.info(
        "Evaluating Gaussian field on %d³ grid (%d voxels, %d Gaussians) …",
        resolution,
        grid_flat.shape[0],
        means.shape[0],
    )

    # ── Density evaluation ──
    volume_flat = evaluate_gaussian_field(
        grid_flat, means, R, inv_s, alpha, g_min, g_max, chunk_size=chunk_size,
    )
    volume = volume_flat.reshape(resolution, resolution, resolution).cpu().numpy()

    # ── Marching cubes ──
    try:
        verts, faces, _normals, _values = marching_cubes(
            volume,
            level=float(isovalue),
            spacing=tuple(float(v) for v in voxel_size),
        )
    except Exception as e:
        logger.error("Marching cubes failed (isovalue=%.4f): %s", isovalue, e)
        return None

    origin = mins.cpu().numpy()
    verts = verts + origin

    logger.info(
        "Marching cubes done: %d vertices, %d faces.", verts.shape[0], faces.shape[0]
    )
    return verts.astype(np.float64), faces.astype(np.int32)


@torch.no_grad()
def export_gaussian_mesh_as_ply(
    means: torch.Tensor,
    scales_exp: torch.Tensor,
    quats: torch.Tensor,
    opacities_sigmoid: torch.Tensor,
    output_path: Path,
    *,
    resolution: int = 128,
    isovalue: float = 1.0,
    padding_factor: float = 0.05,
    culling_sigma: float = 3.0,
    chunk_size: int = 16**3,
) -> Optional[Path]:
    """Extract a mesh from Gaussians and export as PLY.

    Requires *open3d* for mesh I/O and *scikit-image* for marching cubes.

    Returns:
        Path to the written PLY file, or *None* on failure.
    """
    try:
        import open3d as o3d  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "open3d is required to export Gaussian meshes as PLY. "
            "Install with: pip install open3d"
        )
        return None

    result = extract_mesh_from_gaussians(
        means,
        scales_exp,
        quats,
        opacities_sigmoid,
        resolution=resolution,
        isovalue=isovalue,
        padding_factor=padding_factor,
        culling_sigma=culling_sigma,
        chunk_size=chunk_size,
    )
    if result is None:
        return None

    verts, faces = result

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_path), mesh)

    logger.info("Gaussian mesh exported to %s", output_path)
    return output_path
