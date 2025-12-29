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

    method: Literal["tile"] = "tile"
    """Candidate selection strategy. Only 'tile' is implemented."""

    tile_size: int = 16
    """Tile size (pixels) used for screen-space binning."""

    tile_neighbor_radius: int = 1
    """How many neighboring tiles to include (1 => 3x3 neighborhood)."""

    max_gaussians_per_tile: int = 128
    """Cap on Gaussians stored per tile (largest-z pruning happens implicitly)."""

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


def _camera_world_to_camera_matrix(camera: Cameras, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return world->camera 4x4 for a single camera (index 0) on the requested device/dtype."""
    # `camera_to_worlds` is [1, 3, 4] for single camera.
    c2w_3x4 = camera.camera_to_worlds[0].to(device=device, dtype=dtype)
    bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)[None, :]
    c2w = torch.cat([c2w_3x4, bottom], dim=0)  # [4, 4]
    w2c = torch.inverse(c2w)
    return w2c


def _project_points_to_pixels(
    camera: Cameras,
    xyz_world: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points to pixel coordinates using Nerfstudio camera (single cam).

    Returns:
        u: [N] pixel x
        v: [N] pixel y
        z_cam: [N] camera-space z (positive in front)
    """
    device = xyz_world.device
    dtype = xyz_world.dtype
    w2c = _camera_world_to_camera_matrix(camera, device=device, dtype=dtype)  # [4, 4]
    xyz_h = torch.cat([xyz_world, torch.ones_like(xyz_world[..., :1])], dim=-1)  # [N, 4]
    xyz_cam_h = (xyz_h @ w2c.T)  # [N, 4]
    xyz_cam = xyz_cam_h[..., :3]
    x_cam, y_cam, z_cam = xyz_cam.unbind(dim=-1)

    # Intrinsics
    K = camera.get_intrinsics_matrices()[0].to(device=device, dtype=dtype)  # [3, 3]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z_safe = z_cam.clamp_min(eps)
    u = fx * (x_cam / z_safe) + cx
    v = fy * (y_cam / z_safe) + cy
    return u, v, z_cam


def _build_tile_index(
    u: torch.Tensor,
    v: torch.Tensor,
    z_cam: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute tile ids for projected points; returns (tile_id, valid_mask)."""
    in_front = z_cam > 0
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid = in_front & in_img

    tile_x = torch.floor(u / tile_size).to(torch.long)
    tile_y = torch.floor(v / tile_size).to(torch.long)
    num_tiles_x = (width + tile_size - 1) // tile_size
    tile_id = tile_y * num_tiles_x + tile_x
    return tile_id, valid


def _make_tile_table(
    tile_id: torch.Tensor,
    z_cam: torch.Tensor,
    valid: torch.Tensor,
    num_tiles: int,
    max_per_tile: int,
) -> torch.Tensor:
    """Create a [num_tiles, max_per_tile] table of Gaussian indices (padded with -1).

    Implementation detail: we (1) filter to valid, (2) sort by tile_id then z,
    (3) for each tile take up to max_per_tile closest gaussians in z.
    """
    device = tile_id.device
    all_idx = torch.arange(tile_id.shape[0], device=device, dtype=torch.long)
    all_idx = all_idx[valid]
    tile_id = tile_id[valid]
    z_cam = z_cam[valid]

    if all_idx.numel() == 0:
        return torch.full((num_tiles, max_per_tile), -1, device=device, dtype=torch.long)

    # Sort primarily by tile_id, secondarily by z (near first).
    # We do a stable(ish) two-pass: sort by z, then stable sort by tile_id is not guaranteed,
    # so instead encode a composite key. z is float; we quantize it for ordering.
    z_q = torch.clamp((z_cam * 1000.0).to(torch.long), min=0, max=2_000_000_000)
    key = tile_id.to(torch.long) * 2_000_000_001 + z_q
    order = torch.argsort(key)
    all_idx = all_idx[order]
    tile_id = tile_id[order]

    tile_table = torch.full((num_tiles, max_per_tile), -1, device=device, dtype=torch.long)

    # Iterate tiles (<= ~4096 for 512px with tile_size=8/16): cheap enough.
    unique_tiles, counts = torch.unique_consecutive(tile_id, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    for t, s, c in zip(unique_tiles.tolist(), starts.tolist(), counts.tolist()):
        # Fill this tile's row with up to max_per_tile indices.
        end = s + min(c, max_per_tile)
        tile_table[t, : end - s] = all_idx[s:end]

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

    # Candidate selection via screen-space tile binning of Gaussian centers.
    u, v, z_cam = _project_points_to_pixels(camera, means, eps=config.eps)
    tile_id, valid = _build_tile_index(u, v, z_cam, width=W, height=H, tile_size=config.tile_size)
    num_tiles_x = (W + config.tile_size - 1) // config.tile_size
    num_tiles_y = (H + config.tile_size - 1) // config.tile_size
    num_tiles = num_tiles_x * num_tiles_y

    tile_table = _make_tile_table(
        tile_id=tile_id,
        z_cam=z_cam,
        valid=valid,
        num_tiles=num_tiles,
        max_per_tile=config.max_gaussians_per_tile,
    )  # [num_tiles, M]

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

    cand_idx = _gather_neighbor_tiles(
        tile_table=tile_table,
        pixel_tile_id=pix_tile_id[active_idx],
        width=W,
        height=H,
        tile_size=config.tile_size,
        neighbor_radius=config.tile_neighbor_radius,
    )  # [R_active, C]

    # Gather candidate camera depths for hint filtering.
    cand_z = None
    if depth_hint_flat is not None:
        valid_cand = cand_idx >= 0
        safe_idx = torch.where(valid_cand, cand_idx, torch.zeros_like(cand_idx))
        cand_z = z_cam[safe_idx]
        cand_z = torch.where(valid_cand, cand_z, torch.full_like(cand_z, -torch.inf))

    t_active = _ray_ellipsoid_first_hit(
        origins=origins[active_idx],
        directions=directions[active_idx],
        cand_idx=cand_idx,
        means=means,
        inv_scales2=inv_scales2,
        rotmats=rotmats,
        k=config.k,
        eps=config.eps,
        depth_hint=depth_hint_flat[active_idx] if depth_hint_flat is not None else None,
        cand_z_cam=cand_z,
        depth_hint_rel_tol=config.depth_hint_rel_tol,
        depth_hint_abs_tol=config.depth_hint_abs_tol,
    )

    # Scatter back into full image.
    depth_flat = torch.zeros((H * W,), device=origins.device, dtype=origins.dtype)
    depth_flat[active_idx] = t_active
    depth = depth_flat.view(H, W, 1)
    return depth


