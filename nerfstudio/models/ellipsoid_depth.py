"""
Ellipsoid-based depth estimation for Splatfacto / 3D Gaussian Splatting.

Goal
-----
Compute depth as the *first geometric intersection* between camera rays and
ellipsoids derived from 3D Gaussians, instead of the rasterizer's volumetric /
alpha-composited expected depth.

Key constraint (important)
--------------------------
Naively intersecting all rays with all Gaussians is O(R*N) and is not feasible
for typical image sizes and Gaussian counts. This implementation therefore
includes a candidate-selection stage (tile binning in screen space) to limit
the per-ray Gaussian set.

Notes
-----
- Splatfacto stores quaternions in **xyzw** order (see `random_quat_tensor`).
- Scales in Splatfacto are stored in log-space; callers typically pass
  `torch.exp(scales)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from nerfstudio.cameras.cameras import Cameras


@dataclass
class EllipsoidDepthConfig:
    """Config for ellipsoid depth estimation."""

    k: float = 2.0
    """Confidence parameter in (x-mu)^T Sigma^{-1} (x-mu) = k."""

    # NOTE: We intentionally only support gsplat-based binning/sorting.
    # The previous heuristic "tile" mode (center-only projection) was removed because it can disagree
    # with gsplat's true footprint-based tile hits and is harder to tune robustly.

    tile_size: int = 16
    """Tile size (pixels) used for screen-space binning."""

    tile_neighbor_radius: int = 1
    """How many neighboring tiles to include (1 => 3x3 neighborhood)."""

    max_gaussians_per_tile: int = 128
    """Cap on Gaussians stored per tile (largest-z pruning happens implicitly)."""

    ray_chunk_size: int = 8192
    """Number of rays processed per chunk to bound peak GPU memory."""

    use_depth_hint: bool = True
    """If True, filter candidates by proximity to a per-pixel depth hint (if provided)."""

    depth_hint_rel_tol: float = 0.15
    """Relative tolerance around depth_hint: keep z in [d*(1-tol), d*(1+tol)]."""

    depth_hint_abs_tol: float = 0.05
    """Absolute tolerance in scene units added to the depth hint window."""

    eps: float = 1e-8
    """Small epsilon for numeric stability."""


def quat_xyzw_to_rotmat(quat_xyzw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert xyzw quaternion(s) to rotation matrix.

    Args:
        quat_xyzw: [..., 4] quaternion in xyzw order.
        eps: numeric epsilon used for safe normalization.

    Returns:
        rot: [..., 3, 3]
    """
    q = quat_xyzw
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps))
    x, y, z, w = q.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    rot = torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )
    return rot


def _get_gsplat_viewmat(camera_to_world_3x4: torch.Tensor) -> torch.Tensor:
    """Convert Nerfstudio c2w (3x4) to gsplat view matrix (world2camera, 4x4).

    This mirrors `nerfstudio.models.splatfacto.get_viewmat`.
    """
    R = camera_to_world_3x4[:3, :3]  # [3, 3]
    T = camera_to_world_3x4[:3, 3:4]  # [3, 1]
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([1, -1, -1], device=R.device, dtype=R.dtype)[None, :]
    R_inv = R.transpose(0, 1)
    T_inv = -R_inv @ T
    viewmat = torch.zeros((4, 4), device=R.device, dtype=R.dtype)
    viewmat[3, 3] = 1.0
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    return viewmat


def _camera_world_to_camera_matrix(camera: Cameras, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return world->camera 4x4 for a single camera (index 0) on the requested device/dtype."""
    # `camera_to_worlds` is [1, 3, 4] for single camera.
    c2w_3x4 = camera.camera_to_worlds[0].to(device=device, dtype=dtype)
    bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)[None, :]
    c2w = torch.cat([c2w_3x4, bottom], dim=0)  # [4, 4]
    w2c = torch.inverse(c2w)
    return w2c


def _make_tile_table_from_gsplat(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    camera: Cameras,
    width: int,
    height: int,
    tile_size: int,
    max_per_tile: int,
    eps: float,
) -> torch.Tensor:
    """Use gsplat's projection + tile-intersection sorting to build a [num_tiles, max_per_tile] tile table.

    This uses the same kernels as gsplat rasterization:
    - `fully_fused_projection(...)` to get `means2d`, `radii`, `depths`
    - `isect_tiles(...)` to map Gaussians to intersecting tiles (sorted by tile|depth)
    - `isect_offset_encode(...)` to compute per-tile ranges into the sorted intersection list
    """
    import math

    try:
        import gsplat  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"gsplat is required but could not be imported: {e}") from e

    device = means.device
    dtype = means.dtype

    # Prepare viewmats / intrinsics in gsplat conventions.
    viewmat = _get_gsplat_viewmat(camera.camera_to_worlds[0].to(device=device, dtype=dtype))
    viewmats = viewmat[None, None, ...]  # [1, 1, 4, 4] (batch..., C, 4, 4)
    Ks = camera.get_intrinsics_matrices().to(device=device, dtype=dtype)  # [1, 3, 3]
    Ks = Ks[None, ...]  # [1, 1, 3, 3]

    # Project Gaussians to 2D (matches rasterizer behavior).
    # Returns: radii, means2d, depths, conics, compensations
    radii, means2d, depths, _conics, _comp = gsplat.fully_fused_projection(
        means=means[None, ...],  # [1, N, 3]
        covars=None,
        quats=quats[None, ...],  # [1, N, 4]
        scales=scales[None, ...],  # [1, N, 3]
        viewmats=viewmats,  # [1, 1, 4, 4]
        Ks=Ks,  # [1, 1, 3, 3]
        width=width,
        height=height,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        packed=False,
        sparse_grad=False,
        calc_compensations=False,
        camera_model="pinhole",
        opacities=None,
    )
    # Drop batch/camera dims -> [N, 2], [N, 2], [N]
    means2d = means2d[0, 0]
    radii = radii[0, 0]
    depths = depths[0, 0]

    tile_width = int(math.ceil(width / float(tile_size)))
    tile_height = int(math.ceil(height / float(tile_size)))
    num_tiles = tile_width * tile_height

    # Compute tile intersections (sorted by tile|depth by default).
    _tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
        means2d=means2d[None, ...],  # [1, N, 2]
        radii=radii[None, ...],  # [1, N, 2]
        depths=depths[None, ...],  # [1, N]
        tile_size=int(tile_size),
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
        packed=False,
        n_images=None,
        image_ids=None,
        gaussian_ids=None,
    )

    # Offsets per tile into the sorted intersection list.
    # Shape: [I, tile_h, tile_w] where I=1.
    isect_offsets = gsplat.isect_offset_encode(isect_ids=isect_ids, n_images=1, tile_width=tile_width, tile_height=tile_height)
    offsets_flat = isect_offsets.reshape(-1).to(torch.long)  # [num_tiles]

    n_isects = int(flatten_ids.shape[0])
    # End offset is next tile start; last tile ends at n_isects.
    ends_flat = torch.empty_like(offsets_flat)
    ends_flat[:-1] = offsets_flat[1:]
    ends_flat[-1] = n_isects

    # flatten_ids are in [I*N]; for I=1, gaussian_id = flatten_id.
    # For safety, mod by N anyway.
    gaussian_ids_sorted = (flatten_ids.to(torch.long) % means.shape[0]).contiguous()

    tile_table = torch.full((num_tiles, max_per_tile), -1, device=device, dtype=torch.long)
    for t in range(num_tiles):
        lo = int(offsets_flat[t].item())
        hi = int(ends_flat[t].item())
        if hi <= lo:
            continue
        ids = gaussian_ids_sorted[lo:hi]
        if ids.numel() == 0:
            continue
        tile_table[t, : min(max_per_tile, ids.numel())] = ids[:max_per_tile]

    # Return also depths for hint pruning to avoid recomputing; we stash it on the function object.
    _make_tile_table_from_gsplat._last_depths = depths  # type: ignore[attr-defined]
    return tile_table


def _gather_neighbor_tiles(
    tile_table: torch.Tensor,
    pixel_tile_id: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    neighbor_radius: int,
) -> torch.Tensor:
    """For each pixel tile id, gather candidates from neighboring tiles."""
    device = tile_table.device
    num_tiles_x = (width + tile_size - 1) // tile_size
    num_tiles_y = (height + tile_size - 1) // tile_size
    num_tiles = num_tiles_x * num_tiles_y

    tile_x = pixel_tile_id % num_tiles_x
    tile_y = pixel_tile_id // num_tiles_x

    offsets = torch.arange(-neighbor_radius, neighbor_radius + 1, device=device, dtype=torch.long)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(-1)  # [K]
    oy = oy.reshape(-1)  # [K]

    nx = (tile_x[:, None] + ox[None, :]).clamp(0, num_tiles_x - 1)
    ny = (tile_y[:, None] + oy[None, :]).clamp(0, num_tiles_y - 1)
    n_id = ny * num_tiles_x + nx  # [R, K]
    n_id = n_id.clamp(0, num_tiles - 1)

    # Gather [R, K, max_per_tile] then flatten
    gathered = tile_table[n_id]  # [R, K, M]
    return gathered.reshape(gathered.shape[0], -1)  # [R, K*M]


def _ray_ellipsoid_first_hit(
    origins: torch.Tensor,
    directions: torch.Tensor,
    cand_idx: torch.Tensor,
    means: torch.Tensor,
    inv_scales2: torch.Tensor,
    rotmats: torch.Tensor,
    k: float,
    eps: float,
    depth_hint: Optional[torch.Tensor] = None,
    cand_z_cam: Optional[torch.Tensor] = None,
    depth_hint_rel_tol: float = 0.15,
    depth_hint_abs_tol: float = 0.05,
) -> torch.Tensor:
    """Compute per-ray first positive intersection with candidate ellipsoids."""
    device = origins.device
    R = origins.shape[0]

    # Replace invalid candidates (-1) by 0 for indexing, then mask them out.
    valid_cand = cand_idx >= 0
    safe_idx = torch.where(valid_cand, cand_idx, torch.zeros_like(cand_idx))

    mu = means[safe_idx]  # [R, C, 3]
    inv_s2 = inv_scales2[safe_idx]  # [R, C, 3]
    Rt = rotmats[safe_idx].transpose(-1, -2)  # [R, C, 3, 3]

    o = origins[:, None, :]  # [R, 1, 3]
    d = directions[:, None, :]  # [R, 1, 3]
    p = o - mu  # [R, C, 3]

    # Transform into ellipsoid local frame: p' = R^T p, d' = R^T d
    p_local = torch.einsum("rcij,rcj->rci", Rt, p)
    d_local = torch.einsum("rcij,rcj->rci", Rt, d.expand_as(p))

    a = (d_local * d_local * inv_s2).sum(dim=-1)  # [R, C]
    b = 2.0 * (p_local * d_local * inv_s2).sum(dim=-1)  # [R, C]
    c = (p_local * p_local * inv_s2).sum(dim=-1) - k  # [R, C]

    disc = b * b - 4.0 * a * c
    ok = valid_cand & (a > eps) & (disc > 0)

    # Optional hint-based filtering in camera depth (heuristic)
    if depth_hint is not None and cand_z_cam is not None:
        d0 = depth_hint[:, None]
        lo = d0 * (1.0 - depth_hint_rel_tol) - depth_hint_abs_tol
        hi = d0 * (1.0 + depth_hint_rel_tol) + depth_hint_abs_tol
        ok = ok & (cand_z_cam >= lo) & (cand_z_cam <= hi)

    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Prefer the smaller positive root; if it's not positive, try the other.
    t = torch.where(t1 > eps, t1, t2)
    t = torch.where((t > eps) & ok, t, torch.full_like(t, torch.inf))

    # Reduce over candidates
    tmin = t.min(dim=1).values  # [R]
    # Replace inf with max finite for viewer friendliness.
    if torch.isfinite(tmin).any():
        max_finite = torch.max(tmin[torch.isfinite(tmin)])
        tmin = torch.where(torch.isfinite(tmin), tmin, max_finite)
    else:
        tmin = torch.zeros((R,), device=device, dtype=origins.dtype)
    return tmin


@torch.no_grad()
def compute_ellipsoid_depth(
    camera: Cameras,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    depth_hint: Optional[torch.Tensor] = None,
    alpha_mask: Optional[torch.Tensor] = None,
    config: Optional[EllipsoidDepthConfig] = None,
) -> torch.Tensor:
    """Compute ellipsoid-based depth for a single Nerfstudio camera.

    Args:
        camera: Nerfstudio `Cameras` (single camera).
        means: [N, 3] gaussian centers in world coordinates.
        scales: [N, 3] gaussian scales (linear, not log).
        quats: [N, 4] gaussian quaternions (xyzw).
        depth_hint: optional [H, W, 1] depth hint used to filter candidates.
        alpha_mask: optional [H, W, 1] mask; if provided, only pixels with alpha>0 are computed.
        config: EllipsoidDepthConfig.

    Returns:
        depth: [H, W, 1]
    """
    if config is None:
        config = EllipsoidDepthConfig()

    assert camera.shape[0] == 1, "compute_ellipsoid_depth expects a single camera"
    device = means.device
    dtype = means.dtype
    H = int(camera.height.item())
    W = int(camera.width.item())

    # Generate rays for full image (keep_shape => [H, W, 3]).
    rays = camera.generate_rays(camera_indices=0, keep_shape=True)
    origins = rays.origins.to(device=device, dtype=dtype).reshape(-1, 3)
    directions = rays.directions.to(device=device, dtype=dtype).reshape(-1, 3)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(config.eps)

    # Flatten optional masks / hints.
    if alpha_mask is not None:
        alpha_flat = alpha_mask.reshape(-1)
        active = alpha_flat > 0
    else:
        active = torch.ones((H * W,), device=origins.device, dtype=torch.bool)

    depth_hint_flat = None
    if config.use_depth_hint and depth_hint is not None:
        depth_hint_flat = depth_hint.reshape(-1).clamp_min(config.eps)

    # Precompute per-gaussian rotation and inverse scale^2.
    rotmats = quat_xyzw_to_rotmat(quats, eps=config.eps)  # [N, 3, 3]
    inv_scales2 = 1.0 / (scales * scales).clamp_min(config.eps)  # [N, 3]

    num_tiles_x = (W + config.tile_size - 1) // config.tile_size
    num_tiles_y = (H + config.tile_size - 1) // config.tile_size
    num_tiles = num_tiles_x * num_tiles_y

    # Candidate selection via gsplat's tile binning/sorting.
    tile_table = _make_tile_table_from_gsplat(
        means=means,
        scales=scales,
        quats=quats,
        camera=camera,
        width=W,
        height=H,
        tile_size=config.tile_size,
        max_per_tile=config.max_gaussians_per_tile,
        eps=config.eps,
    )

    # Per-gaussian camera z-depth for optional hint pruning (from the same gsplat projection).
    z_cam = getattr(_make_tile_table_from_gsplat, "_last_depths", None)
    if z_cam is None:
        # Should never happen; fallback to no pruning.
        z_cam = torch.full((means.shape[0],), 1e10, device=device, dtype=dtype)

    # Per-pixel tile ids (computed from pixel coordinates).
    px = torch.arange(W, device=origins.device, dtype=torch.long)
    py = torch.arange(H, device=origins.device, dtype=torch.long)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
    pix_tile_x = (grid_x.reshape(-1) // config.tile_size).clamp(0, num_tiles_x - 1)
    pix_tile_y = (grid_y.reshape(-1) // config.tile_size).clamp(0, num_tiles_y - 1)
    pix_tile_id = pix_tile_y * num_tiles_x + pix_tile_x  # [R]

    # Gather candidates for active rays only (huge speed win in sparse regions).
    active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
    if active_idx.numel() == 0:
        return torch.zeros((H, W, 1), device=origins.device, dtype=origins.dtype)

    # Scatter back into full image. We fill in chunks to keep peak memory bounded.
    depth_flat = torch.zeros((H * W,), device=origins.device, dtype=origins.dtype)

    chunk = max(int(config.ray_chunk_size), 1)
    for start in range(0, active_idx.numel(), chunk):
        sel = active_idx[start : start + chunk]

        cand_idx = _gather_neighbor_tiles(
            tile_table=tile_table,
            pixel_tile_id=pix_tile_id[sel],
            width=W,
            height=H,
            tile_size=config.tile_size,
            neighbor_radius=config.tile_neighbor_radius,
        )  # [R_chunk, C]

        # Gather candidate camera depths for hint filtering (chunked).
        cand_z = None
        if depth_hint_flat is not None:
            valid_cand = cand_idx >= 0
            safe_idx = torch.where(valid_cand, cand_idx, torch.zeros_like(cand_idx))
            cand_z = z_cam[safe_idx]
            cand_z = torch.where(valid_cand, cand_z, torch.full_like(cand_z, -torch.inf))

        t_chunk = _ray_ellipsoid_first_hit(
            origins=origins[sel],
            directions=directions[sel],
            cand_idx=cand_idx,
            means=means,
            inv_scales2=inv_scales2,
            rotmats=rotmats,
            k=config.k,
            eps=config.eps,
            depth_hint=depth_hint_flat[sel] if depth_hint_flat is not None else None,
            cand_z_cam=cand_z,
            depth_hint_rel_tol=config.depth_hint_rel_tol,
            depth_hint_abs_tol=config.depth_hint_abs_tol,
        )

        depth_flat[sel] = t_chunk

    depth = depth_flat.view(H, W, 1)
    return depth


