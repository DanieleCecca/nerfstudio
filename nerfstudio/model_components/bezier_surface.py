"""
Bezier surface utilities (lightweight, no SciPy/Matplotlib).

This module is intended for initialization / geometry scaffolding:
- Given a set of 3D points (e.g. COLMAP sparse points) and a semantic label per point,
  generate per-class Bezier surface patches.
- Each patch is a 4x4 Bezier surface (16 control points).

Design goals:
- Clean, dependency-light (torch + stdlib only).
- KNN-based neighborhood selection with configurable k.
- Configurable number of patches per semantic class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch


@dataclass
class BezierPatchGeneratorConfig:
    """Configuration for generating Bezier surface patches from labeled point clouds."""

    num_patches_per_class: int = 1
    """How many patches to generate for each semantic class label."""

    knn_k: int = 64
    """How many neighbors to consider when building a patch around a chosen center.
    Must be >= 16 for a 4x4 patch.
    """

    control_grid_size: int = 4
    """Bezier control grid size per axis. 4 -> 4x4 -> 16 control points."""

    center_sampling: Literal["fps", "random"] = "fps"
    """How to choose patch centers within each semantic class."""

    seed: int = 0
    """Random seed used when center_sampling='random' (and FPS start)."""

    device: Optional[str] = None
    """Optional device for computations (defaults to points' device)."""


class BezierSurfacePatch:
    """Single Bezier surface patch defined by an (m x n) grid of control points (default 4x4)."""

    def __init__(self, control_points: torch.Tensor):
        """
        Args:
            control_points: (m, n, 3) float tensor of control points.
        """
        if control_points.ndim != 3 or control_points.shape[-1] != 3:
            raise ValueError(f"control_points must have shape (m,n,3), got {tuple(control_points.shape)}")
        self.control_points = control_points

    @staticmethod
    def bernstein(i: int, n: int, t: torch.Tensor) -> torch.Tensor:
        """Bernstein polynomial B_{i,n}(t)."""
        # math.comb is exact integer binomial coefficient.
        c = float(math.comb(n, i))
        return c * (t**i) * ((1.0 - t) ** (n - i))

    def sample(self, num_u: int = 20, num_v: int = 20) -> torch.Tensor:
        """
        Sample points on the surface.

        Args:
            num_u: samples along u in [0,1]
            num_v: samples along v in [0,1]

        Returns:
            surface_points: (num_u, num_v, 3)
        """
        cp = self.control_points
        m = int(cp.shape[0]) - 1
        n = int(cp.shape[1]) - 1
        device = cp.device
        dtype = cp.dtype

        u = torch.linspace(0.0, 1.0, steps=num_u, device=device, dtype=dtype)
        v = torch.linspace(0.0, 1.0, steps=num_v, device=device, dtype=dtype)

        # Precompute Bernstein basis for all u,v.
        Bu = torch.stack([self.bernstein(i, m, u) for i in range(m + 1)], dim=0)  # (m+1, num_u)
        Bv = torch.stack([self.bernstein(j, n, v) for j in range(n + 1)], dim=0)  # (n+1, num_v)

        # Evaluate surface: S(u,v) = sum_{i,j} Bu[i,u] * Bv[j,v] * P[i,j]
        # We do this with einsum for clarity.
        # Output shape: (num_u, num_v, 3)
        return torch.einsum("iu,jv,ijc->uvc", Bu, Bv, cp)


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared L2 distances between x (N,3) and y (M,3) -> (N,M)."""
    # (x - y)^2 = x^2 + y^2 - 2xy
    x2 = (x * x).sum(dim=-1, keepdim=True)  # (N,1)
    y2 = (y * y).sum(dim=-1, keepdim=True).transpose(0, 1)  # (1,M)
    xy = x @ y.transpose(0, 1)  # (N,M)
    return torch.clamp(x2 + y2 - 2.0 * xy, min=0.0)


def _fps_indices(points: torch.Tensor, num_samples: int, seed: int = 0) -> torch.Tensor:
    """Simple farthest point sampling (FPS) indices. O(N * num_samples), good for small num_samples."""
    N = int(points.shape[0])
    if num_samples >= N:
        return torch.arange(N, device=points.device)
    g = torch.Generator(device=points.device)
    g.manual_seed(int(seed))
    start = int(torch.randint(low=0, high=N, size=(1,), generator=g, device=points.device).item())

    selected = torch.empty((num_samples,), dtype=torch.long, device=points.device)
    selected[0] = start
    d2 = _pairwise_sq_dists(points, points[start : start + 1]).squeeze(-1)  # (N,)
    for i in range(1, num_samples):
        idx = int(torch.argmax(d2).item())
        selected[i] = idx
        d2 = torch.minimum(d2, _pairwise_sq_dists(points, points[idx : idx + 1]).squeeze(-1))
    return selected


def _knn_indices(points: torch.Tensor, center: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of k nearest neighbors of `center` among `points`."""
    # points: (N,3), center: (3,)
    d2 = ((points - center[None, :]) ** 2).sum(dim=-1)  # (N,)
    k = min(int(k), int(points.shape[0]))
    return torch.topk(d2, k=k, largest=False).indices


def _select_4x4_control_points(neighbors: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    """Select grid_size^2 control points from neighbors by fitting a local 2D frame (PCA) and snapping to a grid."""
    if grid_size != 4:
        raise NotImplementedError("Currently only grid_size=4 (16 control points) is supported.")
    if neighbors.shape[0] < 16:
        raise ValueError("Need at least 16 points to form a 4x4 control grid.")

    pts = neighbors
    device = pts.device
    dtype = pts.dtype

    # Local frame via PCA (use first two principal components as u/v axes).
    mu = pts.mean(dim=0, keepdim=True)
    X = pts - mu
    # covariance 3x3
    C = (X.transpose(0, 1) @ X) / max(1, int(pts.shape[0]) - 1)
    # eigenvectors sorted by eigenvalue ascending; take last two for largest variance
    evals, evecs = torch.linalg.eigh(C)
    u_axis = evecs[:, -1]
    v_axis = evecs[:, -2]

    uv = torch.stack([X @ u_axis, X @ v_axis], dim=-1)  # (K,2)
    u = uv[:, 0]
    v = uv[:, 1]

    # grid targets via quartiles on u and v
    q = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], device=device, dtype=dtype)
    u_q = torch.quantile(u, q)
    v_q = torch.quantile(v, q)

    # For each (i,j), pick nearest 2D point to (u_q[i], v_q[j]) with uniqueness.
    chosen: List[int] = []
    chosen_mask = torch.zeros((pts.shape[0],), dtype=torch.bool, device=device)
    for i in range(4):
        for j in range(4):
            target = torch.tensor([u_q[i], v_q[j]], device=device, dtype=dtype)
            d2 = ((uv - target[None, :]) ** 2).sum(dim=-1)
            # get a few candidates in ascending distance
            cand = torch.argsort(d2)
            picked = None
            for idx in cand.tolist()[:64]:
                if not bool(chosen_mask[idx]):
                    picked = idx
                    break
            if picked is None:
                # fallback: first available
                avail = torch.nonzero(~chosen_mask, as_tuple=False)
                picked = int(avail[0].item())
            chosen_mask[picked] = True
            chosen.append(int(picked))

    cp = pts[torch.tensor(chosen, device=device, dtype=torch.long)].view(4, 4, 3)
    return cp


@torch.no_grad()
def generate_bezier_patches_from_labeled_points(
    points_xyz: torch.Tensor,
    semantic_labels: torch.Tensor,
    config: Optional[BezierPatchGeneratorConfig] = None,
) -> Dict[int, List[BezierSurfacePatch]]:
    """Generate per-class Bezier surface patches from 3D points and integer labels.

    Args:
        points_xyz: (N,3) float tensor of 3D points (COLMAP space as used by the pipeline).
        semantic_labels: (N,) int tensor. 0 is treated as background and skipped.
        config: generator config.

    Returns:
        Dict[label_id, List[BezierSurfacePatch]]
    """
    cfg = config or BezierPatchGeneratorConfig()
    if points_xyz.ndim != 2 or points_xyz.shape[-1] != 3:
        raise ValueError(f"points_xyz must have shape (N,3), got {tuple(points_xyz.shape)}")
    if semantic_labels.ndim != 1 or semantic_labels.shape[0] != points_xyz.shape[0]:
        raise ValueError("semantic_labels must have shape (N,) matching points_xyz.")
    if cfg.knn_k < cfg.control_grid_size * cfg.control_grid_size:
        raise ValueError("knn_k must be >= control_grid_size^2 (need enough points for control points).")
    if cfg.control_grid_size != 4:
        raise NotImplementedError("Currently only control_grid_size=4 (16 control points) is supported.")

    device = points_xyz.device if cfg.device is None else torch.device(cfg.device)
    pts_all = points_xyz.to(device=device)
    labels_all = semantic_labels.to(device=device)

    out: Dict[int, List[BezierSurfacePatch]] = {}
    uniq = torch.unique(labels_all)
    uniq = uniq[uniq > 0]  # skip background
    for lab in uniq.tolist():
        lab_i = int(lab)
        mask = labels_all == lab_i
        pts = pts_all[mask]
        if int(pts.shape[0]) < 16:
            continue

        # Choose patch centers.
        num_patches = min(int(cfg.num_patches_per_class), int(pts.shape[0]))
        if cfg.center_sampling == "fps":
            centers_idx = _fps_indices(pts, num_patches, seed=cfg.seed)
        else:
            g = torch.Generator(device=device)
            g.manual_seed(int(cfg.seed))
            perm = torch.randperm(int(pts.shape[0]), generator=g, device=device)
            centers_idx = perm[:num_patches]

        patches: List[BezierSurfacePatch] = []
        for ci in centers_idx.tolist():
            center = pts[int(ci)]
            nn_idx = _knn_indices(pts, center, k=int(cfg.knn_k))
            neigh = pts[nn_idx]
            cp = _select_4x4_control_points(neigh, grid_size=4)
            patches.append(BezierSurfacePatch(cp))

        out[lab_i] = patches
    return out


def example_usage() -> None:  # pragma: no cover
    """Minimal example (manual run) for sanity checking."""
    # Fake labeled points: two blobs with labels 1 and 2.
    g = torch.Generator().manual_seed(0)
    pts1 = torch.randn((200, 3), generator=g) * 0.1 + torch.tensor([0.0, 0.0, 0.0])
    pts2 = torch.randn((200, 3), generator=g) * 0.1 + torch.tensor([1.0, 0.0, 0.0])
    points = torch.cat([pts1, pts2], dim=0)
    labels = torch.cat([torch.ones((200,), dtype=torch.int64), 2 * torch.ones((200,), dtype=torch.int64)], dim=0)

    cfg = BezierPatchGeneratorConfig(num_patches_per_class=2, knn_k=64, center_sampling="fps", seed=0)
    patches = generate_bezier_patches_from_labeled_points(points, labels, cfg)
    for lab, plist in patches.items():
        print("label", lab, "num_patches", len(plist))
        surf = plist[0].sample(10, 10)
        print("surface sample shape:", tuple(surf.shape))

