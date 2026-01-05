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
from typing import Any, Dict, Optional, Tuple

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class EllipsoidDepthConfig:
    """Config for ellipsoid depth estimation."""

    k: float = 9.0
    """Confidence parameter in (x-mu)^T Sigma^{-1} (x-mu) = k.
    
    Interpretation (for 3D Gaussian):
      k=1   -> ~1σ surface (~39% volume)
      k=4   -> ~2σ surface (~86% volume)  
      k=9   -> ~3σ surface (~99% volume) [RECOMMENDED]
      k=16  -> ~4σ surface
    
    Larger k = bigger ellipsoids = more ray hits but less "tight" surface.
    """

    tile_size: int = 16
    """Tile size (pixels) used for screen-space binning."""

    tile_neighbor_radius: int = 1
    """How many neighboring tiles to include (1 => 3x3 neighborhood)."""

    max_gaussians_per_tile: int = 256
    """Cap on Gaussians stored per tile (largest-z pruning happens implicitly)."""

    ray_chunk_size: int = 8192
    """Number of rays processed per chunk to bound peak GPU memory."""

    use_depth_hint: bool = False
    """If True, filter candidates by proximity to a per-pixel depth hint.
    
    WARNING: This can be overly aggressive and reject valid intersections.
    Only enable if you're sure the rasterizer depth is a good proxy.
    """

    depth_hint_rel_tol: float = 0.5
    """Relative tolerance around depth_hint: keep if |z - hint| < hint * tol."""

    depth_hint_abs_tol: float = 1.0
    """Absolute tolerance in scene units added to the depth hint window."""

    no_hit_value: float = 0.0
    """Value to use for rays that don't hit any ellipsoid.
    
    Options:
      0.0  -> black in depth visualization
      nan  -> will show as distinct color (if colormap handles NaN)
      -1.0 -> can be masked in post-processing
    """

    debug: bool = False
    """If True, print debug statistics about hit rates."""

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
    c2w_3x4 = camera.camera_to_worlds[0].to(device=device, dtype=dtype)
    bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)[None, :]
    c2w = torch.cat([c2w_3x4, bottom], dim=0)  # [4, 4]
    w2c = torch.inverse(c2w)
    return w2c


def _make_tile_table_from_gsplat_meta(
    meta: Dict[str, Any],
    num_gaussians: int,
    max_per_tile: int,
) -> Tuple[torch.Tensor, int, int, int, Dict[str, Any]]:
    """Build per-tile candidate lists using gsplat rasterization meta (no extra gsplat calls).
    
    Returns:
        tile_table: [num_tiles, max_per_tile] tensor of Gaussian indices (-1 = invalid)
        tile_width: number of tiles horizontally
        tile_height: number of tiles vertically
        tile_size: pixel size of each tile
        stats: debug statistics dict
    """
    required_keys = ["tile_width", "tile_height", "tile_size", "flatten_ids", "isect_offsets"]
    missing = [k for k in required_keys if k not in meta]
    if missing:
        raise KeyError(f"gsplat_meta missing required keys: {missing}. Available keys: {list(meta.keys())}")
    
    tile_width = int(meta["tile_width"])
    tile_height = int(meta["tile_height"])
    tile_size = int(meta["tile_size"])

    flatten_ids = meta["flatten_ids"]
    isect_offsets = meta["isect_offsets"]

    # Use first batch/camera if present: [..., C, th, tw] -> [th, tw]
    while isect_offsets.dim() > 2:
        isect_offsets = isect_offsets[0]

    device = flatten_ids.device
    num_tiles = tile_width * tile_height

    offsets_flat = isect_offsets.reshape(-1).to(torch.long)  # [num_tiles]
    n_isects = int(flatten_ids.shape[0])
    ends_flat = torch.empty_like(offsets_flat)
    ends_flat[:-1] = offsets_flat[1:]
    ends_flat[-1] = n_isects

    # gsplat's flatten_ids contains Gaussian indices (may need modulo for packed formats)
    gaussian_ids_sorted = (flatten_ids.to(torch.long) % num_gaussians).contiguous()

    tile_table = torch.full((num_tiles, max_per_tile), -1, device=device, dtype=torch.long)
    
    total_candidates = 0
    non_empty_tiles = 0
    max_per_tile_actual = 0
    
    for t in range(num_tiles):
        lo = int(offsets_flat[t].item())
        hi = int(ends_flat[t].item())
        if hi <= lo:
            continue
        ids = gaussian_ids_sorted[lo:hi]
        if ids.numel() == 0:
            continue
        non_empty_tiles += 1
        count = min(max_per_tile, ids.numel())
        tile_table[t, :count] = ids[:count]
        total_candidates += count
        max_per_tile_actual = max(max_per_tile_actual, ids.numel())

    stats = {
        "num_tiles": num_tiles,
        "non_empty_tiles": non_empty_tiles,
        "total_candidates": total_candidates,
        "max_per_tile_actual": max_per_tile_actual,
        "avg_per_non_empty_tile": total_candidates / max(1, non_empty_tiles),
    }

    return tile_table, tile_width, tile_height, tile_size, stats


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
    depth_hint_rel_tol: float = 0.5,
    depth_hint_abs_tol: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-ray first positive intersection with candidate ellipsoids.
    
    Returns:
        tmin: [R] ray parameter of first hit (inf if no hit)
        hit_mask: [R] bool tensor indicating valid hits
    """
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

    # Quadratic coefficients: a*t^2 + b*t + c = 0
    # where the ellipsoid equation is sum_i (x_i / s_i)^2 = k
    a = (d_local * d_local * inv_s2).sum(dim=-1)  # [R, C]
    b = 2.0 * (p_local * d_local * inv_s2).sum(dim=-1)  # [R, C]
    c = (p_local * p_local * inv_s2).sum(dim=-1) - k  # [R, C]

    disc = b * b - 4.0 * a * c
    ok = valid_cand & (a > eps) & (disc >= 0)

    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a.clamp_min(eps))
    t2 = (-b + sqrt_disc) / (2.0 * a.clamp_min(eps))

    # Prefer the smaller positive root; if it's not positive, try the other.
    t = torch.where(t1 > eps, t1, t2)
    t = torch.where((t > eps) & ok, t, torch.full_like(t, torch.inf))

    # Optional: filter by depth hint (comparing t-values, NOT z_cam)
    if depth_hint is not None:
        d0 = depth_hint[:, None]  # [R, 1]
        lo = d0 * (1.0 - depth_hint_rel_tol) - depth_hint_abs_tol
        hi = d0 * (1.0 + depth_hint_rel_tol) + depth_hint_abs_tol
        # Only keep hits where t is within the hint range
        hint_ok = (t >= lo) & (t <= hi)
        t = torch.where(hint_ok, t, torch.full_like(t, torch.inf))

    # Reduce over candidates
    tmin = t.min(dim=1).values  # [R]
    hit_mask = torch.isfinite(tmin)
    
    return tmin, hit_mask


@torch.no_grad()
def compute_ellipsoid_depth(
    camera: Cameras,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    depth_hint: Optional[torch.Tensor] = None,
    alpha_mask: Optional[torch.Tensor] = None,
    config: Optional[EllipsoidDepthConfig] = None,
    gsplat_meta: Optional[Dict[str, Any]] = None,
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
        gsplat_meta: meta dict from gsplat rasterization (required).

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

    if gsplat_meta is None:
        raise RuntimeError("gsplat_meta is required (pass the meta dict from gsplat rasterization).")

    tile_table, tile_width, tile_height, tile_size, tile_stats = _make_tile_table_from_gsplat_meta(
        meta=gsplat_meta,
        num_gaussians=int(means.shape[0]),
        max_per_tile=config.max_gaussians_per_tile,
    )

    # Per-pixel tile ids (computed from pixel coordinates).
    px = torch.arange(W, device=origins.device, dtype=torch.long)
    py = torch.arange(H, device=origins.device, dtype=torch.long)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
    pix_tile_x = (grid_x.reshape(-1) // tile_size).clamp(0, tile_width - 1)
    pix_tile_y = (grid_y.reshape(-1) // tile_size).clamp(0, tile_height - 1)
    pix_tile_id = pix_tile_y * tile_width + pix_tile_x  # [R]

    # Gather candidates for active rays only (huge speed win in sparse regions).
    active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
    if active_idx.numel() == 0:
        return torch.zeros((H, W, 1), device=origins.device, dtype=origins.dtype)

    # Initialize output with no_hit_value
    depth_flat = torch.full((H * W,), config.no_hit_value, device=origins.device, dtype=origins.dtype)
    hit_mask_flat = torch.zeros((H * W,), device=origins.device, dtype=torch.bool)

    # Debug counters
    total_hits = 0
    total_rays = 0
    total_candidates_checked = 0

    chunk = max(int(config.ray_chunk_size), 1)
    for start in range(0, active_idx.numel(), chunk):
        sel = active_idx[start : start + chunk]
        total_rays += sel.numel()

        cand_idx = _gather_neighbor_tiles(
            tile_table=tile_table,
            pixel_tile_id=pix_tile_id[sel],
            width=W,
            height=H,
            tile_size=tile_size,
            neighbor_radius=config.tile_neighbor_radius,
        )  # [R_chunk, C]

        # Count valid candidates
        total_candidates_checked += (cand_idx >= 0).sum().item()

        t_chunk, hit_chunk = _ray_ellipsoid_first_hit(
            origins=origins[sel],
            directions=directions[sel],
            cand_idx=cand_idx,
            means=means,
            inv_scales2=inv_scales2,
            rotmats=rotmats,
            k=config.k,
            eps=config.eps,
            depth_hint=depth_hint_flat[sel] if depth_hint_flat is not None else None,
            depth_hint_rel_tol=config.depth_hint_rel_tol,
            depth_hint_abs_tol=config.depth_hint_abs_tol,
        )

        # Only write valid hits
        depth_flat[sel] = torch.where(hit_chunk, t_chunk, torch.full_like(t_chunk, config.no_hit_value))
        hit_mask_flat[sel] = hit_chunk
        total_hits += hit_chunk.sum().item()

    # Debug output
    if config.debug:
        hit_rate = 100.0 * total_hits / max(1, total_rays)
        avg_cands = total_candidates_checked / max(1, total_rays)
        CONSOLE.log(
            f"[cyan]Ellipsoid depth stats:[/cyan] "
            f"k={config.k:.1f}, "
            f"hit_rate={hit_rate:.1f}%, "
            f"rays={total_rays}, "
            f"avg_candidates/ray={avg_cands:.1f}, "
            f"tiles: {tile_stats['non_empty_tiles']}/{tile_stats['num_tiles']} non-empty, "
            f"max_per_tile={tile_stats['max_per_tile_actual']}"
        )

    depth = depth_flat.view(H, W, 1)
    return depth
