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

    max_gaussians_per_tile: int = 256
    """Cap on Gaussians stored per tile (only used if method="tile")."""

    screen_filter: bool = True
    """If True (recommended), filter per-tile candidates per-pixel using gsplat's projected
    screen-space Gaussian footprint (means2d/conics/opacities) before doing ray–ellipsoid
    intersection. This avoids tile-wise artifacts where a Gaussian overlaps the tile but not
    the specific pixel."""

    screen_alpha_threshold: float = 1.0 / 255.0
    """Alpha threshold used by gsplat rasterizer (ALPHA_THRESHOLD). Candidates with
    opac*exp(-sigma) below this are ignored for that pixel."""

    output_depth_space: Literal["ray_t", "camera_z"] = "camera_z"
    """Output depth convention:

    - "ray_t": return the ray parameter t for the first ellipsoid hit (meters along the ray in world space).
    - "camera_z": return camera z-depth for the hit point (meters along the camera forward axis).

    Notes:
    - Nerfstudio cameras use an OpenGL-like convention where the camera looks along -Z in camera space,
      so z-depth is computed as `-p_cam[..., 2]`.
    - "camera_z" is directly comparable to gsplat rasterizer depth and DA3 metric depth.
    """

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
    debug: bool = False,
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

    flatten_ids = meta["flatten_ids"]#lista con indici gaussiane
    isect_offsets_orig = meta["isect_offsets"]

    if debug:
        CONSOLE.log(f"[magenta]gsplat_meta shapes: flatten_ids={flatten_ids.shape}, isect_offsets={isect_offsets_orig.shape}[/magenta]")
        CONSOLE.log(f"[magenta]tile_width={tile_width}, tile_height={tile_height}, tile_size={tile_size}[/magenta]")

    # Use first batch/camera if present: [B, C, th, tw] o [B, tile_h, tile_w]-> [th, tw]
    isect_offsets = isect_offsets_orig#ci dice da che indice a che indice vanno i blocchi di gauss
    while isect_offsets.dim() > 2:
        isect_offsets = isect_offsets[0]

    if debug:
        CONSOLE.log(f"[magenta]isect_offsets after squeeze: {isect_offsets.shape}[/magenta]")

    device = flatten_ids.device
    num_tiles = tile_width * tile_height

    # IMPORTANT: gsplat stores isect_offsets as [tile_height, tile_width] in row-major order
    #es [[0, 2, 2],
    #   [3, 5, 6]]->offsets_flat = [0, 2, 2, 3, 5, 6]
    offsets_flat = isect_offsets.reshape(-1).to(torch.long)  # [num_tiles]
    
    if debug:
        CONSOLE.log(f"[magenta]offsets_flat: len={len(offsets_flat)}, expected num_tiles={num_tiles}[/magenta]")
        CONSOLE.log(f"[magenta]offsets_flat first 10: {offsets_flat[:10].tolist()}[/magenta]")
        CONSOLE.log(f"[magenta]offsets_flat last 10: {offsets_flat[-10:].tolist()}[/magenta]")
    
    n_isects = int(flatten_ids.shape[0])
    ends_flat = torch.empty_like(offsets_flat)
    ends_flat[:-1] = offsets_flat[1:]
    ends_flat[-1] = n_isects

    # gsplat's flatten_ids contains Gaussian indices (may need modulo for packed formats)
    # questo perchè l'id potrebbe essere impacchettato(lo è in gsplat) nel seguente modo raw_id = gaussian_id + K * num_gaussians
    #quindi forziamo gli indici ad essere corretti
    gaussian_ids_sorted = (flatten_ids.to(torch.long) % num_gaussians).contiguous()

    if debug:
        CONSOLE.log(f"[magenta]n_isects={n_isects}, num_gaussians={num_gaussians}[/magenta]")
        CONSOLE.log(f"[magenta]gaussian_ids range: min={gaussian_ids_sorted.min().item()}, max={gaussian_ids_sorted.max().item()}[/magenta]")

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

    if debug:
        CONSOLE.log(f"[magenta]Tile table stats: {stats}[/magenta]")

    return tile_table, tile_width, tile_height, tile_size, stats


def _squeeze_meta_first(x: torch.Tensor, min_dim: int) -> torch.Tensor:
    """Repeatedly take the first element of leading batch/camera dims until `x.dim() == min_dim`."""
    while x.dim() > min_dim:
        x = x[0]
    return x


def _filter_candidates_screen_space(
    *,
    sel: torch.Tensor,
    cand_idx: torch.Tensor,
    W: int,
    gsplat_meta: Dict[str, Any],
    alpha_threshold: float,
) -> torch.Tensor:
    """Per-pixel filter for tile candidates, matching gsplat's rasterizer footprint check."""
    if not all(k in gsplat_meta for k in ("means2d", "conics", "opacities")):
        return cand_idx

    means2d = gsplat_meta["means2d"]
    conics = gsplat_meta["conics"]
    opacities = gsplat_meta["opacities"]

    # Squeeze to per-gaussian arrays: means2d [N,2], conics [N,3], opacities [N]
    #così facendo consideriamo solo il primo batch/camera [B, C, N, 2]-> [N, 2]
    means2d = _squeeze_meta_first(means2d, 2)
    #la conica rappresena la forma dell'ellissoide nel piano 2D
    #delta^T * Cov^-1 * delta=costante (Cov^-1 è la precision matrix)
    #conics[i] = (a, b, c) con a=cov_xx^-1, b=cov_xy^-1, c=cov_yy^-1
    #cov simmetrica quindi cov_xy=cov_yx
    conics = _squeeze_meta_first(conics, 2)
    opacities = _squeeze_meta_first(opacities, 1)

    # Pixel coordinates for selected rays (row-major)
    #i=y⋅W+x questo perchè sappiamo che W inizia una nuova riga
    #x=imodW
    #y=⌊i/W⌋
    px = (sel % W).to(dtype=means2d.dtype)
    py = (sel // W).to(dtype=means2d.dtype)

    valid = cand_idx >= 0
    safe_idx = torch.clamp(cand_idx, min=0)

    m = means2d[safe_idx]  # [R, M, 2]
    c = conics[safe_idx]  # [R, M, 3]
    o = opacities[safe_idx]  # [R, M]

    #calcolo le distanze tra il centro della gaussiana e il pixel selezionato
    dx = m[..., 0] - px[:, None]
    dy = m[..., 1] - py[:, None]
    a = c[..., 0]
    b = c[..., 1]
    cc = c[..., 2]
    #ΔTQΔ=[dx​dy​][[ab][​bc]​][[dx][dy​]]=adx2+2bdxdy+cdy2
    sigma = 0.5 * (a * dx * dx + cc * dy * dy) + b * dx * dy
    #contributo = opacity * exp(-sigma)
    alpha = torch.minimum(
        o * torch.exp(-sigma),
        torch.tensor(0.999, device=o.device, dtype=o.dtype),
    )

    keep = valid & (sigma >= 0) & (alpha >= float(alpha_threshold))
    return torch.where(keep, cand_idx, torch.full_like(cand_idx, -1))


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

        # Expand for broadcasting(replicare virtualmente): [R, G, 3]
        o = origins[:, None, :]  # [R, 1, 3]
        d = directions[:, None, :]  # [R, 1, 3]
        mu_exp = mu[None, :, :]  # [1, G, 3]
        P_exp = P[None, :, :, :]  # [1, G, 3, 3]

        p = o - mu_exp  # [R, G, 3]
        dd = d.expand(R, G, 3)  # [R, G, 3]

        # Quadratic coefficients
        #einsum esegue in parallelo le stesse moltiplicazioni vettore–matrice–vettore che faresti in un doppio loop, calcolandole per tutte le coppie di indici specificate.
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

    # Camera transform (camera -> world). We'll use it to convert world hit points back to camera space.
    c2w = camera.camera_to_worlds[0].to(device=device, dtype=dtype)  # [3,4]
    R_c2w = c2w[:, :3]  # [3,3]
    t_c2w = c2w[:, 3]  # [3]

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

            if config.output_depth_space == "camera_z":
                # Convert hit point back to camera space: p_cam = R^T (p_world - t), but for row vectors:
                # p_cam = (p_world - t) @ R, where R = R_c2w
                p_world = origins[sel] + t_chunk[:, None] * directions[sel]  # [R,3]
                p_cam = (p_world - t_c2w[None, :]) @ R_c2w  # [R,3]
                z_depth = -p_cam[:, 2]  # OpenGL convention: forward is -Z
                depth_flat[sel] = torch.where(hit_chunk, z_depth, torch.full_like(z_depth, config.no_hit_value))
            else:
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
            debug=config.debug,
        )

        # Per-pixel tile ids (computed from pixel coordinates).
        px = torch.arange(W, device=origins.device, dtype=torch.long)
        py = torch.arange(H, device=origins.device, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
        pix_tile_x = (grid_x.reshape(-1) // tile_size).clamp(0, tile_width - 1)
        pix_tile_y = (grid_y.reshape(-1) // tile_size).clamp(0, tile_height - 1)
        pix_tile_id = pix_tile_y * tile_width + pix_tile_x  # [R]

        if config.debug:
            CONSOLE.log(f"[magenta]Image size: W={W}, H={H}[/magenta]")
            CONSOLE.log(f"[magenta]Expected tiles: ceil({W}/{tile_size})={-(-W//tile_size)}, ceil({H}/{tile_size})={-(-H//tile_size)}[/magenta]")
            CONSOLE.log(f"[magenta]gsplat tile_width={tile_width}, tile_height={tile_height}[/magenta]")
            CONSOLE.log(f"[magenta]pix_tile_x range: {pix_tile_x.min().item()}-{pix_tile_x.max().item()}[/magenta]")
            CONSOLE.log(f"[magenta]pix_tile_y range: {pix_tile_y.min().item()}-{pix_tile_y.max().item()}[/magenta]")
            CONSOLE.log(f"[magenta]pix_tile_id range: {pix_tile_id.min().item()}-{pix_tile_id.max().item()}[/magenta]")

        chunk = max(int(config.ray_chunk_size), 1)
        for start in range(0, active_idx.numel(), chunk):
            sel = active_idx[start : start + chunk]
            total_rays += sel.numel()

            # Direct lookup: each ray gets candidates from its exact tile
            # gsplat already assigns Gaussians to ALL tiles they overlap, so no need for neighbors
            cand_idx = tile_table[pix_tile_id[sel]]  # [R_chunk, max_per_tile]

            if config.screen_filter:
                cand_idx = _filter_candidates_screen_space(
                    sel=sel,
                    cand_idx=cand_idx,
                    W=W,
                    gsplat_meta=gsplat_meta,
                    alpha_threshold=config.screen_alpha_threshold,
                )

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

            if config.output_depth_space == "camera_z":
                p_world = origins[sel] + t_chunk[:, None] * directions[sel]  # [R,3]
                p_cam = (p_world - t_c2w[None, :]) @ R_c2w  # [R,3]
                z_depth = -p_cam[:, 2]
                depth_flat[sel] = torch.where(hit_chunk, z_depth, torch.full_like(z_depth, config.no_hit_value))
            else:
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
