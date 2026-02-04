"""
SAM2 semantic initialization utilities.

Goal: run SAM2 on a *training* image, then use COLMAP 2D tracks to select the COLMAP 3D points that fall inside the
predicted mask, assigning an integer semantic label per 3D point.

This is intentionally lightweight and defensive:
- SAM2 is treated as an optional dependency (lazy import with actionable install errors).
- Point association prefers COLMAP 2D observations (`points3D_image_ids` + `points3D_points2D_xy`) to avoid re-projecting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SAM2SemanticInitConfig:
    """Configuration for SAM2-based semantic point labeling."""

    model_id: str = "facebook/sam2-hiera-large"
    """HuggingFace model id."""

    image_idx: int = 0
    """Training image index to run SAM2 on."""

    label_id: int = 1
    """Semantic label id assigned to points inside the mask. Background is 0."""

    # Prompts (must provide at least one kind of prompt).
    point_coords: Optional[List[Tuple[float, float]]] = None
    """List of (x,y) points in pixel coordinates (after any dataparser downscale)."""

    point_labels: Optional[List[int]] = None
    """List of prompt labels for point_coords (1=positive, 0=negative)."""

    box_xyxy: Optional[Tuple[float, float, float, float]] = None
    """Optional bounding box (x0,y0,x1,y1) in pixels."""

    mask_distance_px: int = 0
    """If >0, dilate the SAM mask by this pixel radius before point selection."""

    # Prompt-free mode: "segment everything" via a lightweight grid of point prompts.
    auto_grid_stride: int = 32
    """Pixel stride for grid point prompts when no prompts are provided."""

    auto_max_masks: int = 64
    """Maximum number of unique masks to keep in prompt-free mode."""

    auto_min_mask_area: int = 256
    """Minimum mask area (pixels) to keep in prompt-free mode."""

    auto_dedup_iou_thresh: float = 0.9
    """IoU threshold for deduplicating similar masks in prompt-free mode."""

    device: Optional[str] = None
    """Device for SAM2 inference (defaults to 'cuda' if available else 'cpu')."""


class SAM2MaskPredictor:
    """Lazy-loaded SAM2 predictor (HuggingFace weights)."""

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._predictor = None

    def _lazy_load(self) -> None:
        if self._predictor is not None:
            return
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SAM2 is not installed or import failed.\n"
                "Install it (one option) with:\n"
                "  pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'\n"
                "Then the model weights will be fetched from HuggingFace via from_pretrained().\n"
                f"Import error: {e}"
            ) from e

        self._predictor = SAM2ImagePredictor.from_pretrained(self.model_id)

    @torch.no_grad()
    def predict_mask(
        self,
        image_uint8: np.ndarray,
        *,
        point_coords: Optional[Sequence[Tuple[float, float]]] = None,
        point_labels: Optional[Sequence[int]] = None,
        box_xyxy: Optional[Tuple[float, float, float, float]] = None,
    ) -> torch.Tensor:
        """Predict a single binary mask for an input image.

        Args:
            image_uint8: (H,W,3) uint8 RGB image.
            point_coords / point_labels: optional point prompts (pixel space).
            box_xyxy: optional box prompt (pixel space).

        Returns:
            mask: (H,W) bool torch tensor on CPU.
        """
        self._lazy_load()
        assert self._predictor is not None

        if image_uint8.ndim != 3 or image_uint8.shape[2] not in (3, 4):
            raise ValueError(f"Expected image_uint8 to have shape (H,W,3/4), got {image_uint8.shape}")
        if image_uint8.shape[2] == 4:
            image_uint8 = image_uint8[:, :, :3]
        if image_uint8.dtype != np.uint8:
            image_uint8 = image_uint8.astype(np.uint8, copy=False)

        if point_coords is None and box_xyxy is None:
            raise ValueError("SAM2 requires prompts. Provide point_coords (+point_labels) and/or box_xyxy.")
        if point_coords is not None and (point_labels is None or len(point_labels) != len(point_coords)):
            raise ValueError("If point_coords is provided, point_labels must be provided with the same length.")

        self._predictor.set_image(image_uint8)

        # SAM2 docs recommend bfloat16 autocast on CUDA. Keep it defensive for CPU.
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if (self.device.startswith("cuda") and torch.cuda.is_available())
            else torch.autocast("cpu", dtype=torch.float32)
        )

        # The SAM2 predictor API is close to SAM-style predict(point_coords=..., point_labels=..., box=...).
        with torch.inference_mode(), autocast_ctx:
            kwargs: Dict[str, Any] = {}
            if point_coords is not None:
                kwargs["point_coords"] = np.array(point_coords, dtype=np.float32)
                kwargs["point_labels"] = np.array(point_labels, dtype=np.int32)
            if box_xyxy is not None:
                kwargs["box"] = np.array(box_xyxy, dtype=np.float32)

            try:
                masks, _, _ = self._predictor.predict(**kwargs)
            except TypeError:
                # Fallback: some variants accept a single prompt object/dict.
                masks, _, _ = self._predictor.predict(kwargs)

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 returned no masks for the provided prompts.")

        # Prefer the first mask; users can refine prompts if needed.
        mask0 = masks[0]
        if isinstance(mask0, np.ndarray):
            mask_t = torch.from_numpy(mask0)
        else:
            mask_t = torch.as_tensor(mask0)
        mask_t = mask_t.to(dtype=torch.bool, device="cpu")
        if mask_t.ndim == 3:
            # Some APIs return (H,W,1)
            mask_t = mask_t.squeeze(-1)
        if mask_t.ndim != 2:
            raise ValueError(f"Unexpected mask shape from SAM2: {tuple(mask_t.shape)}")
        return mask_t


def _dilate_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Binary dilation via max-pool (CPU)."""
    if radius_px <= 0:
        return mask
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    m = mask[None, None, ...].to(dtype=torch.float32)
    k = 2 * int(radius_px) + 1
    out = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius_px)
    return out[0, 0].to(torch.bool)


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU between two boolean masks (CPU)."""
    inter = torch.logical_and(a, b).sum().item()
    union = torch.logical_or(a, b).sum().item()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


@torch.no_grad()
def _predict_masks_grid_prompts(
    predictor: SAM2MaskPredictor,
    image_uint8: np.ndarray,
    *,
    stride: int,
    max_masks: int,
    min_area: int,
    dedup_iou_thresh: float,
) -> List[torch.Tensor]:
    """Prompt-free 'segment everything' approximation using a grid of positive point prompts.

    This is intentionally simple (and slower than the official automatic mask generator), but keeps the codebase clean
    and avoids extra dependencies. It works well enough as an initialization heuristic.
    """
    if stride <= 0:
        raise ValueError("stride must be > 0")
    H, W = int(image_uint8.shape[0]), int(image_uint8.shape[1])

    predictor._lazy_load()
    assert predictor._predictor is not None
    predictor._predictor.set_image(image_uint8[:, :, :3] if image_uint8.shape[2] == 4 else image_uint8)

    kept: List[torch.Tensor] = []
    # sample points in (x,y) pixel coordinates
    ys = list(range(stride // 2, H, stride))
    xs = list(range(stride // 2, W, stride))
    for y in ys:
        for x in xs:
            if len(kept) >= max_masks:
                return kept

            # Predict a mask from a single positive point.
            try:
                masks, _, _ = predictor._predictor.predict(
                    point_coords=np.array([(float(x), float(y))], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                )
            except TypeError:
                masks, _, _ = predictor._predictor.predict(
                    {
                        "point_coords": np.array([(float(x), float(y))], dtype=np.float32),
                        "point_labels": np.array([1], dtype=np.int32),
                    }
                )
            if masks is None or len(masks) == 0:
                continue

            m0 = masks[0]
            mt = torch.from_numpy(m0) if isinstance(m0, np.ndarray) else torch.as_tensor(m0)
            mt = mt.to(dtype=torch.bool, device="cpu")
            if mt.ndim == 3:
                mt = mt.squeeze(-1)
            if mt.ndim != 2:
                continue

            area = int(mt.sum().item())
            if area < min_area:
                continue

            # Deduplicate by IoU.
            duplicate = False
            for existing in kept:
                if _mask_iou(existing, mt) >= dedup_iou_thresh:
                    duplicate = True
                    break
            if duplicate:
                continue

            kept.append(mt)
    return kept


def _labels_from_labelmap_votes(num_points: int, pidx: torch.Tensor, pixel_labels: torch.Tensor) -> torch.Tensor:
    """Assign a label per point given per-observation pixel labels (CPU). Uses majority vote per point."""
    labels = torch.zeros((num_points,), dtype=torch.int64)
    keep = pixel_labels > 0
    if not bool(keep.any()):
        return labels

    p = pidx[keep].to(torch.int64)
    l = pixel_labels[keep].to(torch.int64)
    max_label = int(l.max().item())
    base = max_label + 1
    key = p * base + l
    order = torch.argsort(key)
    key_s = key[order]
    uniq, counts = torch.unique_consecutive(key_s, return_counts=True)
    p_u = uniq // base
    l_u = uniq % base

    # pick the (point,label) with highest count per point
    idx_desc = torch.argsort(counts, descending=True)
    seen = torch.zeros((num_points,), dtype=torch.bool)
    for j in idx_desc.tolist():
        pi = int(p_u[j].item())
        if not seen[pi]:
            labels[pi] = int(l_u[j].item())
            seen[pi] = True
    return labels

def compute_seed_semantic_labels_from_sam2(
    *,
    train_dataset: Any,
    train_dataparser_outputs: Any,
    config: SAM2SemanticInitConfig,
) -> torch.Tensor:
    """Compute per-COLMAP-3D-point semantic labels using SAM2 on one training image.

    This associates 3D points to the SAM mask using COLMAP 2D tracks:
    - `points3D_image_ids`: (N, M) int64 COLMAP image ids per point (padded with -1)
    - `points3D_points2D_xy`: (N, M, 2) float32 2D coords per observation (already downscaled by the dataparser)

    Returns:
        labels: (N,) int64, 0 for background, config.label_id for in-mask points.
    """
    md: Dict[str, Any] = getattr(train_dataparser_outputs, "metadata", {})
    if "points3D_xyz" not in md:
        raise RuntimeError("Missing 'points3D_xyz' in dataparser metadata; enable COLMAP 3D point loading.")
    if "points3D_image_ids" not in md or "points3D_points2D_xy" not in md:
        raise RuntimeError(
            "Missing COLMAP 2D tracks in metadata. Set ColmapDataParserConfig.max_2D_matches_per_3D_point to -1 or >0."
        )
    if "colmap_im_ids" not in md:
        raise RuntimeError("Missing 'colmap_im_ids' in metadata; update ColmapDataParser to export it.")

    points_image_ids: torch.Tensor = md["points3D_image_ids"]  # (N,M)
    points_xy: torch.Tensor = md["points3D_points2D_xy"]  # (N,M,2)
    colmap_im_ids: torch.Tensor = md["colmap_im_ids"]  # (num_images,)

    num_images = int(colmap_im_ids.shape[0])
    if not (0 <= int(config.image_idx) < num_images):
        raise ValueError(f"config.image_idx={config.image_idx} out of range [0, {num_images-1}]")

    # Load the chosen training image (uint8).
    image_uint8 = train_dataset.get_numpy_image(int(config.image_idx))  # (H,W,3/4) uint8

    labels, _, _ = compute_seed_semantic_labels_and_labelmap_from_sam2(
        train_dataset=train_dataset, train_dataparser_outputs=train_dataparser_outputs, config=config
    )
    return labels

def compute_seed_semantic_labels_and_labelmap_from_sam2(
    *,
    train_dataset: Any,
    train_dataparser_outputs: Any,
    config: SAM2SemanticInitConfig,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Like `compute_seed_semantic_labels_from_sam2`, but also returns the per-pixel label map and the input image.

    Returns:
        labels: (N_points,) int64
        label_map: (H,W) int64 (0=background; >0 are semantic instance ids)
        image_uint8: (H,W,3/4) uint8 training image used for SAM2
    """
    md: Dict[str, Any] = getattr(train_dataparser_outputs, "metadata", {})
    if "points3D_xyz" not in md:
        raise RuntimeError("Missing 'points3D_xyz' in dataparser metadata; enable COLMAP 3D point loading.")
    if "points3D_image_ids" not in md or "points3D_points2D_xy" not in md:
        raise RuntimeError(
            "Missing COLMAP 2D tracks in metadata. Set ColmapDataParserConfig.max_2D_matches_per_3D_point to -1 or >0."
        )
    if "colmap_im_ids" not in md:
        raise RuntimeError("Missing 'colmap_im_ids' in metadata; update ColmapDataParser to export it.")

    points_image_ids: torch.Tensor = md["points3D_image_ids"]  # (N,M)
    points_xy: torch.Tensor = md["points3D_points2D_xy"]  # (N,M,2)
    colmap_im_ids: torch.Tensor = md["colmap_im_ids"]  # (num_images,)

    num_images = int(colmap_im_ids.shape[0])
    if not (0 <= int(config.image_idx) < num_images):
        raise ValueError(f"config.image_idx={config.image_idx} out of range [0, {num_images-1}]")

    image_uint8 = train_dataset.get_numpy_image(int(config.image_idx))  # (H,W,3/4) uint8

    # Run SAM2 -> build label_map
    pred = SAM2MaskPredictor(model_id=config.model_id, device=config.device)
    if config.point_coords is None and config.box_xyxy is None:
        masks = _predict_masks_grid_prompts(
            pred,
            image_uint8,
            stride=int(config.auto_grid_stride),
            max_masks=int(config.auto_max_masks),
            min_area=int(config.auto_min_mask_area),
            dedup_iou_thresh=float(config.auto_dedup_iou_thresh),
        )
        if len(masks) == 0:
            H, W = int(image_uint8.shape[0]), int(image_uint8.shape[1])
            label_map = torch.zeros((H, W), dtype=torch.int64)
            return torch.zeros((points_image_ids.shape[0],), dtype=torch.int64), label_map, image_uint8

        areas = [int(m.sum().item()) for m in masks]
        order = np.argsort(np.array(areas))  # ascending
        H, W = int(masks[0].shape[0]), int(masks[0].shape[1])
        label_map = torch.zeros((H, W), dtype=torch.int64)
        base_label = max(1, int(config.label_id))
        for i, mi in enumerate(order.tolist()):
            label_map[masks[mi]] = base_label + i
    else:
        mask = pred.predict_mask(
            image_uint8,
            point_coords=config.point_coords,
            point_labels=config.point_labels,
            box_xyxy=config.box_xyxy,
        )
        mask = _dilate_mask(mask, int(config.mask_distance_px))
        H, W = int(mask.shape[0]), int(mask.shape[1])
        label_map = torch.zeros((H, W), dtype=torch.int64)
        label_map[mask] = max(1, int(config.label_id))

    # Project 2D tracks for target image -> vote labels per 3D point.
    target_im_id = int(colmap_im_ids[int(config.image_idx)].item())
    if target_im_id < 0:
        raise RuntimeError("Invalid target COLMAP image id (<0).")

    matches = points_image_ids == target_im_id
    pairs = torch.nonzero(matches, as_tuple=False)  # (K,2)
    labels = torch.zeros((points_image_ids.shape[0],), dtype=torch.int64)
    if pairs.numel() == 0:
        return labels, label_map, image_uint8

    pidx = pairs[:, 0]
    oidx = pairs[:, 1]
    xy = points_xy[pidx, oidx]  # (K,2)

    x = torch.round(xy[:, 0]).to(torch.int64)
    y = torch.round(xy[:, 1]).to(torch.int64)
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not bool(in_bounds.any()):
        return labels, label_map, image_uint8

    pidx = pidx[in_bounds]
    x = x[in_bounds]
    y = y[in_bounds]

    pixel_labels = label_map[y, x]
    labels = _labels_from_labelmap_votes(points_image_ids.shape[0], pidx, pixel_labels)
    return labels, label_map, image_uint8

