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
from pathlib import Path
from typing import Dict, List, Literal, Optional

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

    @classmethod
    def bernstein_basis(cls, n: int, t: torch.Tensor) -> torch.Tensor:
        """Stacked Bernstein basis for a given degree n.

        Args:
            n: degree
            t: (...,) tensor in [0,1]

        Returns:
            B: (n+1, ...) where B[i] = B_{i,n}(t)
        """
        return torch.stack([cls.bernstein(i, n, t) for i in range(n + 1)], dim=0)

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
        Bu = self.bernstein_basis(m, u)  # (m+1, num_u)
        Bv = self.bernstein_basis(n, v)  # (n+1, num_v)

        # Evaluate surface: S(u,v) = sum_{i,j} Bu[i,u] * Bv[j,v] * P[i,j]
        # We do this with einsum for clarity.
        # Output shape: (num_u, num_v, 3)
        return torch.einsum("iu,jv,ijc->uvc", Bu, Bv, cp)


def fit_bezier_control_points_from_grid(
    target_points: torch.Tensor,
    m: int,
    n: int,
    u: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Least-squares fit Bezier surface control points to target points on a uniform (u,v) grid.

    This solves, independently per coordinate:
        A @ vec(P) ~= vec(X)
    where A[(i_u,i_v),(i,j)] = B_{i,m}(u_i) * B_{j,n}(v_j).

    Args:
        target_points: (Nu, Nv, 3) points X(u_i,v_j)
        m: Bezier degree in u (control points along u = m+1)
        n: Bezier degree in v (control points along v = n+1)
        u: optional (Nu,) parameter samples in [0,1]. Defaults to uniform linspace.
        v: optional (Nv,) parameter samples in [0,1]. Defaults to uniform linspace.

    Returns:
        control_points: (m+1, n+1, 3)
    """
    if target_points.ndim != 3 or target_points.shape[-1] != 3:
        raise ValueError(f"target_points must have shape (Nu,Nv,3), got {tuple(target_points.shape)}")
    Nu, Nv = int(target_points.shape[0]), int(target_points.shape[1])
    if Nu < (m + 1) or Nv < (n + 1):
        raise ValueError(
            f"Need at least (m+1)x(n+1) samples to fit: got ({Nu},{Nv}) for degrees ({m},{n})."
        )

    device = target_points.device
    dtype = target_points.dtype
    if u is None:
        u = torch.linspace(0.0, 1.0, steps=Nu, device=device, dtype=dtype)
    if v is None:
        v = torch.linspace(0.0, 1.0, steps=Nv, device=device, dtype=dtype)
    assert u is not None
    assert v is not None
    if u.ndim != 1 or u.shape[0] != Nu:
        raise ValueError("u must have shape (Nu,)")
    if v.ndim != 1 or v.shape[0] != Nv:
        raise ValueError("v must have shape (Nv,)")

    Bu = BezierSurfacePatch.bernstein_basis(m, u)  # (m+1, Nu)
    Bv = BezierSurfacePatch.bernstein_basis(n, v)  # (n+1, Nv)

    # A: (Nu*Nv, (m+1)*(n+1))
    A = torch.einsum("iu,jv->uvij", Bu, Bv).reshape(Nu * Nv, (m + 1) * (n + 1))
    Y = target_points.reshape(Nu * Nv, 3)

    # Solve least squares (overdetermined) for vec(P) per coordinate.
    sol = torch.linalg.lstsq(A, Y).solution  # ((m+1)*(n+1), 3)
    return sol.reshape(m + 1, n + 1, 3)


class PairedBezierSurfacePatch:
    """Paired (out/in) Bezier patches with shared (u,v) domain and linear interpolation in-between.

    Given S_out(u,v) and S_in(u,v), define:
        w_r = r/(R-1), r=0..R-1
        S_r(u,v) = (1-w_r) S_out(u,v) + w_r S_in(u,v)

    Because Bezier surfaces are linear in control points, we can equivalently interpolate control points:
        P_r = (1-w_r) P_out + w_r P_in
    """

    def __init__(self, control_points_out: torch.Tensor, control_points_in: torch.Tensor):
        if control_points_out.shape != control_points_in.shape:
            raise ValueError(
                f"control_points_out and control_points_in must have same shape, got "
                f"{tuple(control_points_out.shape)} vs {tuple(control_points_in.shape)}"
            )
        if control_points_out.ndim != 3 or control_points_out.shape[-1] != 3:
            raise ValueError(
                f"control_points must have shape (m,n,3), got {tuple(control_points_out.shape)}"
            )
        self.control_points_out = control_points_out
        self.control_points_in = control_points_in

    def sample_interpolated(self, num_r: int, num_u: int = 20, num_v: int = 20) -> torch.Tensor:
        """Uniformly sample the R interpolated surfaces on the shared (u,v) domain.

        Returns:
            X: (R, num_u, num_v, 3) where X[r,i,j] = S_r(u_i,v_j)
        """
        if num_r < 2:
            raise ValueError("num_r must be >= 2 for paired surface interpolation.")
        cp_out = self.control_points_out
        cp_in = self.control_points_in
        m = int(cp_out.shape[0]) - 1
        n = int(cp_out.shape[1]) - 1
        device = cp_out.device
        dtype = cp_out.dtype

        u = torch.linspace(0.0, 1.0, steps=num_u, device=device, dtype=dtype)
        v = torch.linspace(0.0, 1.0, steps=num_v, device=device, dtype=dtype)
        Bu = BezierSurfacePatch.bernstein_basis(m, u)  # (m+1, num_u)
        Bv = BezierSurfacePatch.bernstein_basis(n, v)  # (n+1, num_v)

        w = torch.linspace(0.0, 1.0, steps=num_r, device=device, dtype=dtype).view(num_r, 1, 1, 1)
        cp = (1.0 - w) * cp_out.unsqueeze(0) + w * cp_in.unsqueeze(0)  # (R, m+1, n+1, 3)

        # Evaluate all R surfaces in batch.
        return torch.einsum("iu,jv,rijc->ruvc", Bu, Bv, cp)


class BezierShellPatch:
    """Closed shell made of paired out/in patches plus 4 side walls sharing border control points.

    The shell is C0-closed (no gaps) by construction because wall patches re-use the exact border
    control points of `control_points_out` and `control_points_in`.
    """

    def __init__(self, control_points_out: torch.Tensor, control_points_in: torch.Tensor):
        if control_points_out.shape != control_points_in.shape:
            raise ValueError(
                f"control_points_out and control_points_in must have same shape, got "
                f"{tuple(control_points_out.shape)} vs {tuple(control_points_in.shape)}"
            )
        if control_points_out.ndim != 3 or control_points_out.shape[-1] != 3:
            raise ValueError(
                f"control_points must have shape (m,n,3), got {tuple(control_points_out.shape)}"
            )
        self.control_points_out = control_points_out
        self.control_points_in = control_points_in

        # Reuse the paired-surface helper for intermediate layers.
        self.paired = PairedBezierSurfacePatch(control_points_out, control_points_in)

    def sample_intermediate_layers(self, num_r: int, num_u: int = 20, num_v: int = 20) -> torch.Tensor:
        """Sample the R interpolated surfaces S_r(u,v) between out and in (including endpoints).

        Returns:
            (R, num_u, num_v, 3)
        """
        return self.paired.sample_interpolated(num_r=num_r, num_u=num_u, num_v=num_v)

    def sample_walls(self, num_r: int, num_u: int = 20, num_v: int = 20) -> Dict[str, torch.Tensor]:
        """Sample the 4 side walls that close the shell.

        We treat each wall as a Bezier surface of degree (edge_degree, 1), where the 2 control points
        along the thickness direction are the out/in edge control points (shared).

        Args:
            num_r: samples along thickness direction (w in [0,1]) to match w_r
            num_u: samples along u for the v-constant walls (top/bottom)
            num_v: samples along v for the u-constant walls (left/right)

        Returns:
            dict with keys: "u0", "u1", "v0", "v1"
            - "u0": (R, num_v, 3) wall for u=0
            - "u1": (R, num_v, 3) wall for u=1
            - "v0": (R, num_u, 3) wall for v=0
            - "v1": (R, num_u, 3) wall for v=1
        """
        if num_r < 2:
            raise ValueError("num_r must be >= 2 for shell walls.")
        cp_out = self.control_points_out
        cp_in = self.control_points_in

        # Wall at u=0 and u=1: parameterized by (v, w)
        cp_u0 = torch.stack([cp_out[0, :, :], cp_in[0, :, :]], dim=1)  # (n+1, 2, 3)
        cp_u1 = torch.stack([cp_out[-1, :, :], cp_in[-1, :, :]], dim=1)  # (n+1, 2, 3)
        wall_u0 = BezierSurfacePatch(cp_u0).sample(num_u=num_v, num_v=num_r)  # (Nv, R, 3)
        wall_u1 = BezierSurfacePatch(cp_u1).sample(num_u=num_v, num_v=num_r)  # (Nv, R, 3)

        # Wall at v=0 and v=1: parameterized by (u, w)
        cp_v0 = torch.stack([cp_out[:, 0, :], cp_in[:, 0, :]], dim=1)  # (m+1, 2, 3)
        cp_v1 = torch.stack([cp_out[:, -1, :], cp_in[:, -1, :]], dim=1)  # (m+1, 2, 3)
        wall_v0 = BezierSurfacePatch(cp_v0).sample(num_u=num_u, num_v=num_r)  # (Nu, R, 3)
        wall_v1 = BezierSurfacePatch(cp_v1).sample(num_u=num_u, num_v=num_r)  # (Nu, R, 3)

        # Transpose to make thickness dimension first: (R, edge_samples, 3)
        return {
            "u0": wall_u0.permute(1, 0, 2).contiguous(),
            "u1": wall_u1.permute(1, 0, 2).contiguous(),
            "v0": wall_v0.permute(1, 0, 2).contiguous(),
            "v1": wall_v1.permute(1, 0, 2).contiguous(),
        }


def sample_paired_bezier_surfaces(
    control_points_out: torch.Tensor,
    control_points_in: torch.Tensor,
    *,
    num_r: int,
    num_u: int = 20,
    num_v: int = 20,
) -> torch.Tensor:
    """Sample interpolated Bezier surfaces between paired out/in control points (batched).

    This is a batched equivalent of :meth:`PairedBezierSurfacePatch.sample_interpolated`.

    Args:
        control_points_out: (m+1, n+1, 3) or (B, m+1, n+1, 3)
        control_points_in:  same shape as control_points_out
        num_r: number of layers along thickness (w) including endpoints
        num_u: samples along u
        num_v: samples along v

    Returns:
        X: (B, R, num_u, num_v, 3) float tensor
    """
    if num_r < 2:
        raise ValueError("num_r must be >= 2 for paired surface interpolation.")
    if control_points_out.shape != control_points_in.shape:
        raise ValueError("control_points_out and control_points_in must have the same shape.")
    if control_points_out.ndim == 3:
        cp_out = control_points_out.unsqueeze(0)
        cp_in = control_points_in.unsqueeze(0)
    elif control_points_out.ndim == 4:
        cp_out = control_points_out
        cp_in = control_points_in
    else:
        raise ValueError(
            f"control_points_out must have shape (m,n,3) or (B,m,n,3), got {tuple(control_points_out.shape)}"
        )
    if cp_out.shape[-1] != 3:
        raise ValueError("control points must have last dim = 3.")

    m = int(cp_out.shape[-3]) - 1
    n = int(cp_out.shape[-2]) - 1
    device = cp_out.device
    dtype = cp_out.dtype

    u = torch.linspace(0.0, 1.0, steps=int(num_u), device=device, dtype=dtype)
    v = torch.linspace(0.0, 1.0, steps=int(num_v), device=device, dtype=dtype)
    Bu = BezierSurfacePatch.bernstein_basis(m, u)  # (m+1, U)
    Bv = BezierSurfacePatch.bernstein_basis(n, v)  # (n+1, V)

    # cp_out/cp_in: (B,m+1,n+1,3)
    w = torch.linspace(0.0, 1.0, steps=int(num_r), device=device, dtype=dtype).view(int(num_r), 1, 1, 1, 1)
    cp = (1.0 - w) * cp_out.unsqueeze(0) + w * cp_in.unsqueeze(0)  # (R,B,m+1,n+1,3)

    # Evaluate all R surfaces in batch: -> (R,B,U,V,3) then permute -> (B,R,U,V,3)
    X_rbuvc = torch.einsum("iu,jv,rbijc->rbuvc", Bu, Bv, cp)
    return X_rbuvc.permute(1, 0, 2, 3, 4).contiguous()


def sample_bezier_surfaces(
    control_points: torch.Tensor,
    *,
    num_u: int = 20,
    num_v: int = 20,
) -> torch.Tensor:
    """Batch-evaluate single Bezier surfaces on a uniform (u,v) grid.

    This is the single-surface counterpart of :func:`sample_paired_bezier_surfaces`
    and is used by the open-mode reparameterization path.

    Args:
        control_points: (m+1, n+1, 3) or (B, m+1, n+1, 3)
        num_u: number of uniform samples along u in [0, 1]
        num_v: number of uniform samples along v in [0, 1]

    Returns:
        X: (B, num_u, num_v, 3) sampled surface points.
    """
    if control_points.ndim == 3:
        cp = control_points.unsqueeze(0)
    elif control_points.ndim == 4:
        cp = control_points
    else:
        raise ValueError(
            f"control_points must have shape (m,n,3) or (B,m,n,3), got {tuple(control_points.shape)}"
        )
    if cp.shape[-1] != 3:
        raise ValueError("control points must have last dim = 3.")

    m = int(cp.shape[-3]) - 1
    n = int(cp.shape[-2]) - 1
    device = cp.device
    dtype = cp.dtype

    u = torch.linspace(0.0, 1.0, steps=int(num_u), device=device, dtype=dtype)
    v = torch.linspace(0.0, 1.0, steps=int(num_v), device=device, dtype=dtype)
    Bu = BezierSurfacePatch.bernstein_basis(m, u)  # (m+1, U)
    Bv = BezierSurfacePatch.bernstein_basis(n, v)  # (n+1, V)

    return torch.einsum("iu,jv,bijc->buvc", Bu, Bv, cp)


def bezier_shell_topo_losses_from_samples(
    X: torch.Tensor,
    *,
    eps: float = 1e-6,
    delta: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute Xing and thickness losses from sampled shell layers.

    Implements the discrete Jacobian determinant penalty:
        J = d_u · (d_v × d_w)
        L_xing = mean(ReLU(eps - J))

    and the minimum-thickness penalty:
        L_thick = mean(ReLU(delta - ||X_{r+1}-X_r||))

    Args:
        X: (R,U,V,3) or (B,R,U,V,3) samples of interpolated layers.
        eps: margin for orientation-preserving volume (prevents near-degenerate volumes).
        delta: desired minimum thickness between adjacent layers (0 disables thickness penalty).

    Returns:
        Dict with keys: "xing", "thick"
    """
    if X.ndim == 4:
        Xb = X.unsqueeze(0)
    elif X.ndim == 5:
        Xb = X
    else:
        raise ValueError(f"X must have shape (R,U,V,3) or (B,R,U,V,3), got {tuple(X.shape)}")
    if Xb.shape[-1] != 3:
        raise ValueError("X must have last dim = 3.")

    B, R, U, V, _ = Xb.shape
    device = Xb.device
    dtype = Xb.dtype

    # Thickness loss (needs R>=2).
    if R >= 2 and float(delta) > 0.0:
        d = torch.linalg.norm(Xb[:, 1:, :, :, :] - Xb[:, :-1, :, :, :], dim=-1)  # (B,R-1,U,V)
        thick = torch.clamp(torch.as_tensor(float(delta), device=device, dtype=dtype) - d, min=0.0).mean()
    else:
        thick = torch.zeros((), device=device, dtype=dtype)

    # Xing loss (needs internal indices in r,u,v -> R>=3, U>=3, V>=3).
    if R >= 3 and U >= 3 and V >= 3:
        # Internal grid only, matching the paper notation for central differences.
        du = Xb[:, 1:-1, 2:, 1:-1, :] - Xb[:, 1:-1, :-2, 1:-1, :]  # (B,R-2,U-2,V-2,3)
        dv = Xb[:, 1:-1, 1:-1, 2:, :] - Xb[:, 1:-1, 1:-1, :-2, :]  # (B,R-2,U-2,V-2,3)
        dw = Xb[:, 2:, 1:-1, 1:-1, :] - Xb[:, :-2, 1:-1, 1:-1, :]  # (B,R-2,U-2,V-2,3)

        J = torch.sum(du * torch.cross(dv, dw, dim=-1), dim=-1)  # (B,R-2,U-2,V-2)
        xing = torch.clamp(torch.as_tensor(float(eps), device=device, dtype=dtype) - J, min=0.0).mean()
    else:
        xing = torch.zeros((), device=device, dtype=dtype)

    return {"xing": xing, "thick": thick}

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


def _create_planar_control_points(neighbors: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    """Create a regular planar grid of Bezier control points via PCA.

    Instead of snapping to existing point cloud points, this builds a truly
    independent grid of control points that lie on the best-fit plane of the
    input neighbourhood.  The grid covers the full extent of the projected
    points along the two principal axes of variance.

    Args:
        neighbors: (K, 3) 3D points in the local neighbourhood.
        grid_size: number of control points per axis (default 4 → 4×4 = 16 CPs).

    Returns:
        cp: (grid_size, grid_size, 3) planar control points.
    """
    pts = neighbors
    device = pts.device
    dtype = pts.dtype

    mu = pts.mean(dim=0)  # (3,)
    X = pts - mu[None, :]  # (K, 3)

    C = (X.transpose(0, 1) @ X) / max(1, int(pts.shape[0]) - 1)
    _evals, evecs = torch.linalg.eigh(C)
    u_axis = evecs[:, -1]  # largest-variance direction
    v_axis = evecs[:, -2]  # second-largest

    proj_u = X @ u_axis  # (K,)
    proj_v = X @ v_axis  # (K,)

    u_lin = torch.linspace(float(proj_u.min()), float(proj_u.max()), grid_size, device=device, dtype=dtype)
    v_lin = torch.linspace(float(proj_v.min()), float(proj_v.max()), grid_size, device=device, dtype=dtype)

    uu, vv = torch.meshgrid(u_lin, v_lin, indexing="ij")  # (G, G)
    cp = (
        mu[None, None, :]
        + uu[:, :, None] * u_axis[None, None, :]
        + vv[:, :, None] * v_axis[None, None, :]
    )
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
            cp = _create_planar_control_points(neigh, grid_size=int(cfg.control_grid_size))
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


# ---------------------------------------------------------------------------
# Mesh export utilities
# ---------------------------------------------------------------------------

def _grid_triangles(num_u: int, num_v: int):
    """Return (F, 3) int32 triangle indices for a (num_u x num_v) vertex grid."""
    import numpy as np

    faces = []
    for i in range(num_u - 1):
        for j in range(num_v - 1):
            a = i * num_v + j
            b = (i + 1) * num_v + j
            c = i * num_v + (j + 1)
            d = (i + 1) * num_v + (j + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])
    return np.asarray(faces, dtype=np.int32)


@torch.no_grad()
def export_bezier_patches_as_ply(
    control_points: torch.Tensor,
    output_dir: Path,
    *,
    num_u: int = 40,
    num_v: int = 40,
    control_points_in: Optional[torch.Tensor] = None,
    num_r: int = 5,
    prefix: str = "bezier_patch",
) -> List[Path]:
    """Export Bezier surface patches as triangle-mesh PLY files.

    For each patch in the batch the surface is sampled on a uniform (u, v) grid
    and triangulated into a mesh.  When *control_points_in* is provided (shell
    mode), the outer surface, the inner surface, and the full shell (all R
    interpolated layers merged) are exported per patch.

    Requires Open3D for mesh construction and PLY I/O.

    Args:
        control_points: (S, m+1, n+1, 3) or (m+1, n+1, 3) outer / open CPs.
        output_dir: destination directory (created if missing).
        num_u: uniform samples along u for the mesh.
        num_v: uniform samples along v for the mesh.
        control_points_in: optional (same shape) inner CPs for shell mode.
        num_r: interpolated layers between out/in (shell mode, >= 2).
        prefix: filename prefix for the exported PLY files.

    Returns:
        List of Path objects for the written files (empty if open3d is missing).
    """
    import logging

    try:
        import open3d as o3d  # type: ignore[import-untyped]
    except ImportError:
        logging.error(
            "open3d is required to export Bezier patches as PLY meshes. "
            "Install it with: pip install open3d"
        )
        return []

    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faces = _grid_triangles(num_u, num_v)
    saved: List[Path] = []

    def _save_mesh(verts_np: np.ndarray, faces_np: np.ndarray, name: str) -> Path:
        p = output_dir / name
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts_np.astype(np.float64)),
            triangles=o3d.utility.Vector3iVector(faces_np.astype(np.int32)),
        )
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(p), mesh)
        saved.append(p)
        return p

    # Ensure batch dimension
    cp = control_points
    if cp.ndim == 3:
        cp = cp.unsqueeze(0)

    X = sample_bezier_surfaces(cp, num_u=num_u, num_v=num_v)  # (S, Nu, Nv, 3)
    S = int(X.shape[0])

    all_verts: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    offset = 0
    for s in range(S):
        v = X[s].detach().cpu().numpy().reshape(-1, 3)
        suffix = "" if control_points_in is None else "_outer"
        _save_mesh(v, faces, f"{prefix}_{s:03d}{suffix}.ply")
        all_verts.append(v)
        all_faces.append(faces + offset)
        offset += v.shape[0]

    if S > 1:
        suffix = "" if control_points_in is None else "_outer"
        _save_mesh(
            np.concatenate(all_verts, axis=0),
            np.concatenate(all_faces, axis=0),
            f"{prefix}_all{suffix}.ply",
        )

    # ── Shell mode: inner surfaces + interpolated layers ──
    if control_points_in is not None:
        cp_in = control_points_in
        if cp_in.ndim == 3:
            cp_in = cp_in.unsqueeze(0)

        X_in = sample_bezier_surfaces(cp_in, num_u=num_u, num_v=num_v)
        for s in range(S):
            v = X_in[s].detach().cpu().numpy().reshape(-1, 3)
            _save_mesh(v, faces, f"{prefix}_{s:03d}_inner.ply")

        if num_r >= 2:
            X_layers = sample_paired_bezier_surfaces(
                cp, cp_in, num_r=num_r, num_u=num_u, num_v=num_v,
            )  # (S, R, Nu, Nv, 3)
            for s in range(S):
                lv: List[np.ndarray] = []
                lf: List[np.ndarray] = []
                off = 0
                for r in range(int(X_layers.shape[1])):
                    v = X_layers[s, r].detach().cpu().numpy().reshape(-1, 3)
                    lv.append(v)
                    lf.append(faces + off)
                    off += v.shape[0]
                _save_mesh(
                    np.concatenate(lv, axis=0),
                    np.concatenate(lf, axis=0),
                    f"{prefix}_{s:03d}_shell.ply",
                )

    return saved

