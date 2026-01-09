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
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class EllipsoidDepthConfig:
    """Config for ellipsoid depth estimation."""

    method: Literal["tile", "bruteforce"] = "tile"
    """Candidate selection method:
    
    - "tile": Use gsplat's tile binning for fast candidate selection (O(R*C) where C << N).
              Requires gsplat_meta from rasterization.
    - "bruteforce": Test ALL Gaussians for each ray (O(R*N)). Slow but exact.
                    Does not require gsplat_meta.
    """

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
    """Tile size (pixels) used for screen-space binning (only used if method="tile")."""

    tile_neighbor_radius: int = 1
    """How many neighboring tiles to include (only used if method="tile")."""

    max_gaussians_per_tile: int = 256
    """Cap on Gaussians stored per tile (only used if method="tile")."""

    ray_chunk_size: int = 8192
    """Number of rays processed per chunk to bound peak GPU memory."""

    gauss_chunk_size: int = 4096
    """Number of Gaussians processed per chunk in bruteforce mode to bound peak GPU memory."""

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


def _xyzw_to_wxyz(quats_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternions from xyzw (Nerfstudio) to wxyz (gsplat) order."""
    return torch.stack([quats_xyzw[..., 3], quats_xyzw[..., 0], quats_xyzw[..., 1], quats_xyzw[..., 2]], dim=-1)


def _get_precision_matrices(quats_xyzw: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Compute precision matrices (Σ⁻¹) using gsplat's quat_scale_to_covar_preci.
    
    Args:
        quats_xyzw: [N, 4] quaternions in xyzw order (Nerfstudio convention).
        scales: [N, 3] scales (linear, not log).
    
    Returns:
        precis: [N, 3, 3] precision matrices.
    """
    from gsplat import quat_scale_to_covar_preci
    
    # gsplat expects wxyz order
    quats_wxyz = _xyzw_to_wxyz(quats_xyzw)
    
    _, precis = quat_scale_to_covar_preci(
        quats_wxyz,
        scales,
        compute_covar=False,
        compute_preci=True,
        triu=False,
    )
    assert precis is not None
    return precis  # [N, 3, 3]



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
    tile_width: int,
    tile_height: int,
    neighbor_radius: int,
) -> torch.Tensor:
    """For each pixel tile id, gather candidates from neighboring tiles.
    
    Args:
        tile_table: [num_tiles, max_per_tile] tensor of Gaussian indices
        pixel_tile_id: [R] tile id for each pixel/ray
        tile_width: number of tiles horizontally (from gsplat_meta)
        tile_height: number of tiles vertically (from gsplat_meta)
        neighbor_radius: how many neighboring tiles to include
    """
    device = tile_table.device
    num_tiles = tile_width * tile_height

    tile_x = pixel_tile_id % tile_width
    tile_y = pixel_tile_id // tile_width

    offsets = torch.arange(-neighbor_radius, neighbor_radius + 1, device=device, dtype=torch.long)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    ox = ox.reshape(-1)  # [K]
    oy = oy.reshape(-1)  # [K]

    nx = (tile_x[:, None] + ox[None, :]).clamp(0, tile_width - 1)
    ny = (tile_y[:, None] + oy[None, :]).clamp(0, tile_height - 1)
    n_id = ny * tile_width + nx  # [R, K]
    n_id = n_id.clamp(0, num_tiles - 1)

    # Gather [R, K, max_per_tile] then flatten
    gathered = tile_table[n_id]  # [R, K, M]
    return gathered.reshape(gathered.shape[0], -1)  # [R, K*M]


def _ray_ellipsoid_first_hit(
    origins: torch.Tensor,
    directions: torch.Tensor,
    cand_idx: torch.Tensor,
    means: torch.Tensor,
    precis: torch.Tensor,
    k: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-ray first positive intersection with candidate ellipsoids (tile mode).
    
    The ellipsoid is defined as (x - μ)^T P (x - μ) = k, where P = Σ⁻¹ is the precision matrix.
    
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
    P = precis[safe_idx]  # [R, C, 3, 3]

    o = origins[:, None, :]  # [R, 1, 3]
    d = directions[:, None, :]  # [R, 1, 3]
    p = o - mu  # [R, C, 3]
    dd = d.expand_as(p)  # [R, C, 3]

    # Quadratic coefficients using precision matrix:
    # For ray o + t*d intersecting ellipsoid (x-μ)^T P (x-μ) = k:
    #   a = d^T P d,  b = 2 p^T P d,  c = p^T P p - k
    a = torch.einsum("rci,rcij,rcj->rc", dd, P, dd)  # [R, C]
    b = 2.0 * torch.einsum("rci,rcij,rcj->rc", p, P, dd)  # [R, C]
    c = torch.einsum("rci,rcij,rcj->rc", p, P, p) - k  # [R, C]

    disc = b * b - 4.0 * a * c
    ok = valid_cand & (a > eps) & (disc >= 0)

    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a.clamp_min(eps))
    t2 = (-b + sqrt_disc) / (2.0 * a.clamp_min(eps))

    # Prefer the smaller positive root; if it's not positive, try the other.
    t = torch.where(t1 > eps, t1, t2)
    t = torch.where((t > eps) & ok, t, torch.full_like(t, torch.inf))

    # Reduce over candidates
    tmin = t.min(dim=1).values  # [R]
    hit_mask = torch.isfinite(tmin)
    
    return tmin, hit_mask


def _ray_ellipsoid_first_hit_bruteforce(
    origins: torch.Tensor,
    directions: torch.Tensor,
    means: torch.Tensor,
    precis: torch.Tensor,
    k: float,
    eps: float,
    gauss_chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-ray first positive intersection with ALL ellipsoids (bruteforce).
    
    The ellipsoid is defined as (x - μ)^T P (x - μ) = k, where P = Σ⁻¹ is the precision matrix.
    
    This tests every ray against every Gaussian (O(R*N)), but processes Gaussians in chunks
    to avoid OOM.
    
    Returns:
        tmin: [R] ray parameter of first hit (inf if no hit)
        hit_mask: [R] bool tensor indicating valid hits
    """
    device = origins.device
    R = origins.shape[0]
    N = means.shape[0]

    # Initialize with inf (no hit)
    tmin = torch.full((R,), torch.inf, device=device, dtype=origins.dtype)

    # Process Gaussians in chunks to bound memory
    for g_start in range(0, N, gauss_chunk_size):
        g_end = min(g_start + gauss_chunk_size, N)
        
        mu = means[g_start:g_end]  # [G, 3]
        P = precis[g_start:g_end]  # [G, 3, 3]
        G = mu.shape[0]

        # Expand for broadcasting: [R, G, 3]
        o = origins[:, None, :]  # [R, 1, 3]
        d = directions[:, None, :]  # [R, 1, 3]
        mu_exp = mu[None, :, :]  # [1, G, 3]
        P_exp = P[None, :, :, :]  # [1, G, 3, 3]

        p = o - mu_exp  # [R, G, 3]
        dd = d.expand(R, G, 3)  # [R, G, 3]

        # Quadratic coefficients
        a = torch.einsum("rgi,rgij,rgj->rg", dd, P_exp.expand(R, G, 3, 3), dd)  # [R, G]
        b = 2.0 * torch.einsum("rgi,rgij,rgj->rg", p, P_exp.expand(R, G, 3, 3), dd)  # [R, G]
        c = torch.einsum("rgi,rgij,rgj->rg", p, P_exp.expand(R, G, 3, 3), p) - k  # [R, G]

        disc = b * b - 4.0 * a * c
        ok = (a > eps) & (disc >= 0)

        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        t1 = (-b - sqrt_disc) / (2.0 * a.clamp_min(eps))
        t2 = (-b + sqrt_disc) / (2.0 * a.clamp_min(eps))

        # Prefer the smaller positive root
        t = torch.where(t1 > eps, t1, t2)
        t = torch.where((t > eps) & ok, t, torch.full_like(t, torch.inf))

        # Reduce over this chunk of Gaussians and update global min
        t_chunk_min = t.min(dim=1).values  # [R]
        tmin = torch.minimum(tmin, t_chunk_min)

    hit_mask = torch.isfinite(tmin)
    return tmin, hit_mask


@torch.no_grad()
def compute_ellipsoid_depth(
    camera: Cameras,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
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
        alpha_mask: optional [H, W, 1] mask; if provided, only pixels with alpha>0 are computed.
        config: EllipsoidDepthConfig.
        gsplat_meta: meta dict from gsplat rasterization (required for method="tile").

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
    N = means.shape[0]

    # Generate rays for full image (keep_shape => [H, W, 3]).
    rays = camera.generate_rays(camera_indices=0, keep_shape=True)
    origins = rays.origins.to(device=device, dtype=dtype).reshape(-1, 3)
    directions = rays.directions.to(device=device, dtype=dtype).reshape(-1, 3)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(config.eps)

    # Flatten optional mask.
    if alpha_mask is not None:
        alpha_flat = alpha_mask.reshape(-1)
        active = alpha_flat > 0
    else:
        active = torch.ones((H * W,), device=origins.device, dtype=torch.bool)

    # Compute precision matrices using gsplat's optimized function.
    precis = _get_precision_matrices(quats, scales)  # [N, 3, 3]

    # Gather candidates for active rays only.
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
    tile_stats = None

    if config.method == "bruteforce":
        # ========== BRUTEFORCE: test all N Gaussians for each ray ==========
        chunk = max(int(config.ray_chunk_size), 1)
        for start in range(0, active_idx.numel(), chunk):
            sel = active_idx[start : start + chunk]
            total_rays += sel.numel()
            total_candidates_checked += sel.numel() * N

            t_chunk, hit_chunk = _ray_ellipsoid_first_hit_bruteforce(
                origins=origins[sel],
                directions=directions[sel],
                means=means,
                precis=precis,
                k=config.k,
                eps=config.eps,
                gauss_chunk_size=config.gauss_chunk_size,
            )

            depth_flat[sel] = torch.where(hit_chunk, t_chunk, torch.full_like(t_chunk, config.no_hit_value))
            hit_mask_flat[sel] = hit_chunk
            total_hits += hit_chunk.sum().item()

    else:
        # ========== TILE: use gsplat's tile binning for candidate selection ==========
        if gsplat_meta is None:
            raise RuntimeError("gsplat_meta is required for method='tile' (pass the meta dict from gsplat rasterization).")

        tile_table, tile_width, tile_height, tile_size, tile_stats = _make_tile_table_from_gsplat_meta(
            meta=gsplat_meta,
            num_gaussians=N,
            max_per_tile=config.max_gaussians_per_tile,
        )

        # Per-pixel tile ids (computed from pixel coordinates).
        px = torch.arange(W, device=origins.device, dtype=torch.long)
        py = torch.arange(H, device=origins.device, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
        pix_tile_x = (grid_x.reshape(-1) // tile_size).clamp(0, tile_width - 1)
        pix_tile_y = (grid_y.reshape(-1) // tile_size).clamp(0, tile_height - 1)
        pix_tile_id = pix_tile_y * tile_width + pix_tile_x  # [R]

        chunk = max(int(config.ray_chunk_size), 1)
        for start in range(0, active_idx.numel(), chunk):
            sel = active_idx[start : start + chunk]
            total_rays += sel.numel()

            cand_idx = _gather_neighbor_tiles(
                tile_table=tile_table,
                pixel_tile_id=pix_tile_id[sel],
                tile_width=tile_width,
                tile_height=tile_height,
                neighbor_radius=config.tile_neighbor_radius,
            )  # [R_chunk, C]

            # Count valid candidates
            total_candidates_checked += (cand_idx >= 0).sum().item()

            t_chunk, hit_chunk = _ray_ellipsoid_first_hit(
                origins=origins[sel],
                directions=directions[sel],
                cand_idx=cand_idx,
                means=means,
                precis=precis,
                k=config.k,
                eps=config.eps,
            )

            depth_flat[sel] = torch.where(hit_chunk, t_chunk, torch.full_like(t_chunk, config.no_hit_value))
            hit_mask_flat[sel] = hit_chunk
            total_hits += hit_chunk.sum().item()

    # Debug output
    if config.debug:
        hit_rate = 100.0 * total_hits / max(1, total_rays)
        avg_cands = total_candidates_checked / max(1, total_rays)
        if config.method == "bruteforce":
            CONSOLE.log(
                f"[cyan]Ellipsoid depth stats (bruteforce):[/cyan] "
                f"k={config.k:.1f}, "
                f"hit_rate={hit_rate:.1f}%, "
                f"rays={total_rays}, "
                f"gaussians={N}, "
                f"total_intersections={total_candidates_checked}"
            )
        elif tile_stats is not None:
            CONSOLE.log(
                f"[cyan]Ellipsoid depth stats (tile):[/cyan] "
                f"k={config.k:.1f}, "
                f"hit_rate={hit_rate:.1f}%, "
                f"rays={total_rays}, "
                f"avg_candidates/ray={avg_cands:.1f}, "
                f"tiles: {tile_stats['non_empty_tiles']}/{tile_stats['num_tiles']} non-empty, "
                f"max_per_tile={tile_stats['max_per_tile_actual']}"
            )

    depth = depth_flat.view(H, W, 1)
    return depth
