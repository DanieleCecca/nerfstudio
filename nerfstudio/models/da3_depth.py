"""
Depth Anything 3 (DA3) metric depth integration.

This module provides a lightweight wrapper around Depth Anything 3 for computing
metric depth maps from an RGB image, intended for use in Nerfstudio's viewer.

Target model: DA3METRIC-LARGE (ByteDance-Seed / Depth-Anything-3).

Notes / caveats
---------------
- DA3 is a monocular model: depth is inferred from the rendered RGB, not from
  the original dataset image.
- This is intended for *visualization* in the viewer. It can be slow.
- The Depth-Anything-3 Python API may change; we therefore keep this wrapper
  defensive and emit actionable error messages if imports fail.

Reference:
- Repo: https://github.com/ByteDance-Seed/Depth-Anything-3
- Model card: https://huggingface.co/depth-anything/DA3METRIC-LARGE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DA3MetricConfig:
    """Configuration for DA3 metric depth inference."""

    model_id: str = "depth-anything/da3metric-large"
    """HuggingFace model id for DA3 metric depth.

    Note: HF ids are lowercase in the DA3 docs/model card.
    """

    max_side: int = 384
    """Downscale so that max(H,W) <= max_side before inference (speed/memory)."""

    use_half: bool = True
    """Use fp16 inference on CUDA (if available)."""


class DA3MetricDepthEstimator:
    """Lazy-loaded DA3 metric depth estimator."""

    def __init__(self, config: Optional[DA3MetricConfig] = None, device: Optional[torch.device] = None):
        self.config = config or DA3MetricConfig()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        try:
            # Basic usage (model card): DepthAnything3.from_pretrained(...)
            from depth_anything_3.api import DepthAnything3  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Depth-Anything-3 is not installed or import failed. "
                "Install it with the optional dependency (see pyproject.toml) or via:\n"
                "  pip install 'depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git'\n"
                f"Import error: {e}"
            ) from e

        model = DepthAnything3.from_pretrained(self.config.model_id)
        model = model.to(device=self.device)
        if self.device.type == "cuda" and self.config.use_half:
            model = model.half()
        self._model = model

    @torch.no_grad()
    def infer_metric_depth(self, rgb: torch.Tensor, focal_px: float) -> torch.Tensor:
        """Infer metric depth from an RGB image.

        Args:
            rgb: [H, W, 3] float in [0,1]
            focal_px: focal length in pixels (use mean of fx,fy).

        Returns:
            depth: [H, W, 1] metric depth (float32).
        """
        self._lazy_load()
        assert self._model is not None

        H, W, _ = rgb.shape
        img = rgb.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        img = img.to(
            device=self.device,
            dtype=torch.float16 if (self.device.type == "cuda" and self.config.use_half) else torch.float32,
        )

        # Resize for speed.
        scale = 1.0
        if max(H, W) > self.config.max_side:
            scale = self.config.max_side / float(max(H, W))
        if scale != 1.0:
            new_h = max(1, int(round(H * scale)))
            new_w = max(1, int(round(W * scale)))
            img_in = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            img_in = img

        # DepthAnything3.inference accepts numpy/PIL/paths. We pass a single tensor image.
        # We convert to uint8 CPU for compatibility with the public API.
        img_uint8 = torch.clamp(img_in[0].permute(1, 2, 0) * 255.0, 0, 255).to(torch.uint8).cpu().numpy()

        pred_obj = self._model.inference([img_uint8])
        pred = torch.from_numpy(pred_obj.depth[0]).to(device=self.device, dtype=torch.float32)  # [h,w]

        # DA3METRIC-LARGE model card:
        # metric_depth = focal * net_output / 300.
        metric = (pred * float(focal_px)) / 300.0

        # Upsample back to original resolution.
        metric = metric[None, None, ...]  # [1,1,h,w]
        if metric.shape[-2:] != (H, W):
            metric = F.interpolate(metric, size=(H, W), mode="bilinear", align_corners=False)

        depth = metric.permute(0, 2, 3, 1)[0].to(dtype=torch.float32)  # [H,W,1]
        return depth


_GLOBAL_DA3: Dict[Tuple[str, int, bool, str], DA3MetricDepthEstimator] = {}


def get_da3_metric_estimator(
    model_id: str,
    max_side: int,
    use_half: bool,
    device: torch.device,
) -> DA3MetricDepthEstimator:
    """Get a cached estimator keyed by (model_id, max_side, use_half, device)."""
    key = (model_id, int(max_side), bool(use_half), str(device))
    if key not in _GLOBAL_DA3:
        _GLOBAL_DA3[key] = DA3MetricDepthEstimator(
            config=DA3MetricConfig(model_id=model_id, max_side=max_side, use_half=use_half),
            device=device,
        )
    return _GLOBAL_DA3[key]


