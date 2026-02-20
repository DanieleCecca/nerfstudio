# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import dataclasses
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from pytorch_msssim import SSIM
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from nerfstudio.model_components.bezier_surface import (
    BezierShellPatch,
    BezierPatchGeneratorConfig,
    PairedBezierSurfacePatch,
    bezier_shell_topo_losses_from_samples,
    export_bezier_patches_as_ply,
    fit_bezier_control_points_from_grid,
    generate_bezier_patches_from_labeled_points,
    sample_bezier_surfaces,
    sample_paired_bezier_surfaces,
)
from nerfstudio.model_components.gaussian_mesh import export_gaussian_mesh_as_ply


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class SplatfactoModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: SplatfactoModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    use_absgrad: bool = True
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    depth_mode: Literal["rasterizer", "ellipsoid", "both"] = "rasterizer"
    """Which depth to output in the `depth` channel.

    - "rasterizer": use gsplat's RGB+ED depth (alpha-composited expected depth)
    - "ellipsoid": use geometric ray–ellipsoid first-hit depth (slower; more stable across views)
    - "both": output `depth` from rasterizer and also output `depth_ellipsoid`
    """

    ellipsoid_depth_method: Literal["tile", "bruteforce"] = "tile"
    """Candidate selection method for ellipsoid depth:
    
    - "tile": Use gsplat's tile binning for fast candidate selection (recommended).
    - "bruteforce": Test ALL Gaussians for each ray. Slow but exact (O(R*N)).
    """
    ellipsoid_depth_k: float = 9.0
    """Confidence parameter k for the ellipsoid surface: (x-mu)^T Sigma^{-1} (x-mu) = k.
    
    Interpretation for a 3D Gaussian:
      k=1  -> ~1σ (thin, may miss)
      k=4  -> ~2σ (reasonable)
      k=9  -> ~3σ (recommended - covers 99% of Gaussian volume)
      k=16 -> ~4σ (very thick ellipsoids)
    """
    ellipsoid_depth_tile_size: int = 16
    """Tile size (pixels) for candidate selection (only used if method="tile")."""
    ellipsoid_depth_max_gaussians_per_tile: int = 256
    """Max Gaussians stored per tile (only used if method="tile")."""
    ellipsoid_depth_gauss_chunk_size: int = 4096
    """Gaussians per chunk in bruteforce mode (only used if method="bruteforce")."""
    ellipsoid_depth_output_space: Literal["ray_t", "camera_z"] = "camera_z"
    """Output space for ellipsoid depth:

    - "ray_t": ray parameter t along world rays (meters along the ray)
    - "camera_z": camera z-depth (meters along camera forward axis; comparable to gsplat rasterizer depth and DA3)
    """
    ellipsoid_depth_debug: bool = True
    """Print debug statistics for ellipsoid depth (hit rate, candidates per ray, etc.)."""

    da3_depth_enabled: bool = False
    """If True, compute Depth Anything 3 metric depth from the rendered RGB and output it as `depth_da3` (eval only)."""

    da3_model_id: str = "depth-anything/DA3METRIC-LARGE"
    """HuggingFace model id for DA3 metric depth."""

    da3_max_side: int = 384
    """Downscale rendered RGB so max(H,W)<=da3_max_side for DA3 inference (speed/memory)."""

    da3_use_half: bool = True
    """Use fp16 inference for DA3 on CUDA (if available)."""

    # --- Segmentation-aware seed filtering (applies to classic init path) ---
    filter_seeds_by_segmentation: bool = False
    """If True and ``metadata['seed_semantic_labels']`` is available, remove background
    seed points (label == 0) even when ``bezier_init_enabled`` is False (classic init).
    The Gaussian model is then trained only on the object of interest.
    Combine with a per-pixel mask in the DataManager for pixel-wise masked loss.
    """

    # --- Bezier-surface init from semantically-labeled COLMAP seed points (optional) ---
    bezier_init_enabled: bool = False
    """If True and `metadata['seed_semantic_labels']` is provided, initialize Gaussians from Bezier surface patches
    built from COLMAP seed points per semantic class.
    """

    bezier_surface_mode: Literal["open", "closed", "both"] = "open"
    """Which Bezier geometry to use for Gaussian initialization:

    - "open": use only the outer Bezier surface S_out(u,v) (r=0).
      When reparam, pruning, topo-loss, or attach-loss are enabled, a thin shell
      (out/in pair) is still constructed behind the scenes so those features work.
    - "closed": use only the interior/inner part of the shell (S_1..S_{R-1}); requires paired out/in
    - "both": use the full shell stack (S_0..S_{R-1}); requires paired out/in
    """

    bezier_num_u: int = 20
    """Nu: number of uniform samples along u for each Bezier patch."""

    bezier_num_v: int = 20
    """Nv: number of uniform samples along v for each Bezier patch."""

    bezier_texture_size: int = 8
    """Size T of the per-patch texture grid (T x T x 4 RGBA). Color at (u,v) is texture_i[u_idx, v_idx]."""

    bezier_rho: float = 20.0
    """Global density/overlap parameter rho (divides tangential distances)."""

    bezier_alpha: float = 1.0
    """Thickness parameter alpha for the normal scale sigma_n = alpha / rho."""

    bezier_knn_k: int = 64
    """KNN neighborhood size used when building each Bezier patch (must be >= 16)."""

    bezier_num_patches_per_class: int = 1
    """How many Bezier patches to generate per semantic class label."""

    bezier_center_sampling: Literal["fps", "random"] = "fps"
    """How to choose patch centers within each semantic class."""

    bezier_seed: int = 0
    """Random seed used for patch center sampling."""

    bezier_num_r: int = 5
    """R: number of interpolated surfaces between S_out and S_in (must be >= 2 when closed mode is enabled)."""

    bezier_closed_thickness: float = 0.02
    """Thickness (in world units) between S_out and S_in.
    Implementation detail: we compute a target inner surface by offsetting sampled outer points along the
    outer surface normal, then fit a Bezier surface (same degree) in least squares.
    """

    bezier_closed_include_walls: bool = False
    """If True, also initialize Gaussians on the 4 side walls that close the shell (in addition to intermediate layers)."""

    # --- Bezier shell topology regularization (optional) ---
    bezier_topo_loss_enabled: bool = False
    """If True, add Xing/thickness regularization terms on the paired Bezier shell surfaces."""

    bezier_topo_lambda_xing: float = 0.0
    """Weight for the Xing loss term (orientation-preserving Jacobian determinant margin)."""

    bezier_topo_lambda_thick: float = 0.0
    """Weight for the minimum-thickness regularizer."""

    bezier_topo_eps: float = 1e-6
    """Margin epsilon for Xing loss: penalize when J < eps."""

    bezier_topo_delta: float = 0.0
    """Minimum thickness delta for thickness loss (0 disables)."""

    bezier_open_cp_l2_lambda: float = 0.0
    """Weight for L2 regularizer on open Bezier control points (reduces excessive deformations)."""

    # --- Bezier shell <-> Gaussian attachment (optional) ---
    bezier_attach_loss_enabled: bool = False
    """If True, add an attachment loss tying (a subset of) Gaussian means to points sampled on the Bezier shell.

    Practical note: Splatfacto can densify/prune Gaussians, which changes ordering/identity. This attachment is therefore
    applied only while the Gaussian set is unchanged (typically early training, before refinement).
    """

    bezier_attach_lambda: float = 0.0
    """Weight for the attachment loss."""

    bezier_attach_stop_step: int = 500
    """Stop applying attachment loss after this training step (to avoid issues once densification/pruning begins)."""

    # --- Bezier reparameterization (mu/scale are functions of Bezier control points) ---
    bezier_reparam_enabled: bool = False
    """If True, reparameterize Gaussian means/scales as functions of the paired Bezier shell control points.

    This makes the main rendering loss backpropagate to the Bezier control points. Practical limitation: the standard
    densification/pruning strategies assume learnable per-Gaussian means/scales; therefore refinement is disabled when
    this mode is enabled.
    """

    # --- Adaptive surface-level pruning for Bezier shells ---
    bezier_surface_pruning_enabled: bool = False
    """If True, run adaptive pruning at the *surface* level (Bezier shell patches) instead of gsplat densify/prune.

    Surfaces are deactivated (and their associated Gaussians are excluded from rendering/loss) based on opacity,
    area, and shell thickness. (Color and AABB IoU redundancy are reserved for future use.)
    """

    bezier_prune_every: int = 100
    """How often (in steps) to run surface pruning."""

    bezier_prune_start_step: int = 500
    """Start surface pruning after this step (gives time for stabilization)."""

    bezier_prune_tau_area: float = 0.0
    """Small-area pruning threshold. 0 disables area pruning."""

    bezier_prune_tau_color: float = 0.0
    """Reserved: neighbor color similarity (Euclidean). Not used by current pruning."""

    bezier_prune_tau_iou: float = 0.9
    """Reserved: redundant-overlap threshold on 3D AABB IoU. Not used by current pruning."""

    bezier_prune_tau_opacity_start: float = 0.01
    """Initial low-opacity pruning threshold (adaptive schedule)."""

    bezier_prune_tau_opacity_end: float = 0.001
    """Final low-opacity pruning threshold (adaptive schedule)."""

    bezier_prune_tau_opacity_max_steps: int = 10000
    """Steps over which tau_opacity is annealed from start to end."""

    bezier_prune_tau_thick_start: float = 0.002
    """Initial collapsed-shell pruning threshold (adaptive schedule)."""

    bezier_prune_tau_thick_end: float = 0.0005
    """Final collapsed-shell pruning threshold (adaptive schedule)."""

    bezier_prune_tau_thick_max_steps: int = 10000
    """Steps over which tau_thick is annealed from start to end."""

    bezier_prune_area_num_u: int = 20
    """U samples used to estimate surface area and thickness for pruning."""

    bezier_prune_area_num_v: int = 20
    """V samples used to estimate surface area and thickness for pruning."""

    # --- SAM2 semantic init (optional; executed at train startup in VanillaPipeline) ---
    sam2_semantic_init_enabled: bool = False
    """If True, run SAM2 at pipeline init to compute semantic labels for COLMAP seed points."""

    sam2_model_id: str = "facebook/sam2-hiera-large"
    """HuggingFace model id for SAM2."""

    sam2_init_image_idx: int = 0
    """Training image index to run SAM2 on."""

    sam2_label_id: int = 1
    """Base label id. If prompts are provided, points inside the mask get this label.
    If no prompts are provided, prompt-free mode assigns labels sam2_label_id, sam2_label_id+1, ... per discovered object.
    """

    # Text-prompt mode (GroundingDINO + SAM2). If sam2_text_prompts is non-empty, VanillaPipeline prefers this path
    # and will (optionally) run segmentation across all training images and vote 3D point labels.
    sam2_text_prompts: Optional[List[str]] = None
    """Optional list of text prompts. Example: ["car", "person"]. Each prompt gets label id sam2_label_id+i."""

    sam2_groundingdino_model_id: str = "IDEA-Research/grounding-dino-base"
    """HuggingFace Hub repo id for GroundingDINO (auto-download config+checkpoint)."""

    sam2_groundingdino_revision: Optional[str] = None
    """Optional HuggingFace revision (branch/tag/commit)."""

    sam2_groundingdino_config_filename: Optional[str] = None
    """Optional override: exact config filename inside the HF repo (auto-picked if None)."""

    sam2_groundingdino_checkpoint_filename: Optional[str] = None
    """Optional override: exact checkpoint filename inside the HF repo (auto-picked if None)."""

    # Legacy explicit-path mode (still supported, but you shouldn't need it).
    sam2_groundingdino_config_path: Optional[str] = None
    """(Legacy) Path to GroundingDINO config .py file."""

    sam2_groundingdino_checkpoint_path: Optional[str] = None
    """(Legacy) Path to GroundingDINO checkpoint .pth file."""

    sam2_groundingdino_box_threshold: float = 0.30
    """GroundingDINO box threshold."""

    sam2_groundingdino_text_threshold: float = 0.25
    """GroundingDINO text threshold."""

    sam2_groundingdino_max_boxes_per_prompt: int = 8
    """Max boxes per prompt per image (runtime bound)."""

    sam2_segment_all_train_images: bool = True
    """If True (recommended), run text-prompt segmentation on all training images and vote 3D point labels."""

    sam2_segment_image_indices: Optional[List[int]] = None
    """Optional explicit list of dataset image indices to segment (overrides sam2_segment_all_train_images)."""

    sam2_segmentation_output_dir: str = "data/grounded_sam2"
    """Directory to save per-image segmentation outputs (labelmap + overlay + metadata)."""

    sam2_save_labelmap_npy: bool = True
    """If True, save labelmap_{idx}.npy with exact integer labels per pixel."""

    sam2_save_per_prompt_masks: bool = False
    """If True, save per-prompt binary masks as PNG (can be large)."""

    # Prompts (optional). If both point prompts and box are None, we run prompt-free "segment everything".
    sam2_point_coords: Optional[List[Tuple[float, float]]] = None
    """List of (x,y) point prompts in pixel coordinates (after any dataparser downscale)."""

    sam2_point_labels: Optional[List[int]] = None
    """List of prompt labels for point coords (1=positive, 0=negative). Must match sam2_point_coords length."""

    sam2_box_xyxy: Optional[Tuple[float, float, float, float]] = None
    """Optional bounding box prompt (x0,y0,x1,y1) in pixels."""

    sam2_mask_distance_px: int = 0
    """If >0, dilate the SAM mask by this pixel radius before point selection."""

    # Prompt-free mode knobs.
    sam2_auto_grid_stride: int = 32
    """Pixel stride for grid point prompts when no prompts are provided."""

    sam2_auto_max_masks: int = 64
    """Maximum number of unique masks (objects) to keep in prompt-free mode."""

    sam2_auto_min_mask_area: int = 256
    """Minimum mask area (pixels) to keep in prompt-free mode."""

    sam2_auto_dedup_iou_thresh: float = 0.9
    """IoU threshold for deduplicating similar masks in prompt-free mode."""

    sam2_device: Optional[str] = None
    """Optional device override for SAM2 inference (e.g., 'cuda' or 'cpu')."""

    export_end_of_training_outputs: bool = False
    """If True, export per-training-camera outputs (rgb / depth_da3 / depth_ellipsoid) when training finishes."""

    export_end_of_training_dirname: str = "end_of_training_outputs"
    """Directory name (inside the experiment base dir) where end-of-training outputs are written."""

    export_end_of_training_num_cameras: int = 10
    """How many training cameras to export at end of training (sampled randomly without replacement)."""

    export_bezier_meshes: bool = False
    """If True, export Bezier surface patches as triangle-mesh PLY files at the end of training.
    Files are saved under ``<export_end_of_training_dirname>/bezier_meshes/``.
    Uses the same (num_u, num_v) sampling resolution as the Gaussian placement grid.
    """

    export_gaussian_mesh: bool = False
    """If True, extract a triangle mesh from the Gaussian field via volumetric
    density evaluation + marching cubes at end of training.  Requires
    ``scikit-image`` and ``open3d``.
    """

    gaussian_mesh_resolution: int = 128
    """Grid resolution per axis for the volumetric density evaluation (e.g. 128 or 256)."""

    gaussian_mesh_isovalue: float = 1.0
    """Iso-surface threshold τ for marching cubes."""

    gaussian_mesh_culling_sigma: float = 3.0
    """Culling radius in multiples of σ for the per-chunk Gaussian AABB test."""

    gaussian_mesh_chunk_size: int = 16**3
    """Grid points per evaluation chunk (lower = less GPU memory)."""

    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""
    strategy: Literal["default", "mcmc"] = "default"
    """The default strategy will be used if strategy is not specified. Other strategies, e.g. mcmc, can be used."""
    max_gs_num: int = 1_000_000
    """Maximum number of GSs. Default to 1_000_000."""
    noise_lr: float = 5e5
    """MCMC samping noise learning rate. Default to 5e5."""
    mcmc_opacity_reg: float = 0.01
    """Regularization term for opacity in MCMC strategy. Only enabled when using MCMC strategy"""
    mcmc_scale_reg: float = 0.01
    """Regularization term for scale in MCMC strategy. Only enabled when using MCMC strategy"""

    loss: Literal["rgb", "depth"] = "rgb"
    """Which main loss to use.

    - "rgb": original Splatfacto loss, L1(gt_rgb, pred_rgb) + SSIM(gt_rgb, pred_rgb)
    - "depth": replace the L1 term with L1(DA3_depth(gt_rgb), expected_depth(pred)), keep SSIM on RGB
    """


class SplatfactoModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SplatfactoModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)

        # Locals initialized for type-checkers; they will be assigned by one of the init paths below.
        means: Optional[torch.nn.Parameter] = None
        scales: Optional[torch.nn.Parameter] = None
        quats: Optional[torch.nn.Parameter] = None
        features_dc: Optional[torch.nn.Parameter] = None
        features_rest: Optional[torch.nn.Parameter] = None
        num_points: int = 0

        bezier_init_done = False
        # Stored only if bezier_topo_loss_enabled and we successfully built closed shells.
        topo_cp_out_list: List[torch.Tensor] = []
        topo_cp_in_list: List[torch.Tensor] = []
        attach_shell_idx_list: List[torch.Tensor] = []
        attach_r_idx_list: List[torch.Tensor] = []
        attach_u_idx_list: List[torch.Tensor] = []
        attach_v_idx_list: List[torch.Tensor] = []
        reparam_shell_idx_list: List[torch.Tensor] = []
        reparam_r_idx_list: List[torch.Tensor] = []
        reparam_u_idx_list: List[torch.Tensor] = []
        reparam_v_idx_list: List[torch.Tensor] = []
        prune_surface_idx_list: List[torch.Tensor] = []
        # Open-mode: trainable control points and per-Gaussian → patch/u/v mapping.
        open_cp_list: List[torch.Tensor] = []
        open_patch_idx_list: List[torch.Tensor] = []
        open_u_idx_list: List[torch.Tensor] = []
        open_v_idx_list: List[torch.Tensor] = []

        # Optional: initialize Gaussians from Bezier surface samples (requires semantic labels for seed points).
        meta = self.kwargs.get("metadata", {}) if isinstance(self.kwargs, dict) else {}
        seed_sem = meta.get("seed_semantic_labels", None) if isinstance(meta, dict) else None
        if (
            self.config.bezier_init_enabled
            and self.seed_points is not None
            and (not self.config.random_init)
            and seed_sem is not None
        ):
            try:
                seed_xyz = torch.as_tensor(self.seed_points[0]).detach().cpu().float()
                seed_rgb = torch.as_tensor(self.seed_points[1]).detach().cpu()
                seed_labels = torch.as_tensor(seed_sem).detach().cpu().to(torch.int64)

                # Use only segmented/object seed points (label > 0) for Bezier surface construction.
                # This keeps the Bezier init focused on actual object surfaces and avoids background clutter.
                if int(seed_labels.shape[0]) != int(seed_xyz.shape[0]):
                    raise ValueError(
                        f"seed_semantic_labels length mismatch: labels={int(seed_labels.shape[0])} vs seed_xyz={int(seed_xyz.shape[0])}"
                    )
                obj_mask = seed_labels > 0
                if not bool(obj_mask.any()):
                    raise ValueError("No segmented seed points (all labels are 0); skipping Bezier init.")
                seed_xyz_obj = seed_xyz[obj_mask]
                seed_rgb_obj = seed_rgb[obj_mask] if seed_rgb.numel() > 0 and int(seed_rgb.shape[0]) == int(seed_xyz.shape[0]) else seed_rgb
                seed_labels_obj = seed_labels[obj_mask]

                patch_cfg = BezierPatchGeneratorConfig(
                    num_patches_per_class=int(self.config.bezier_num_patches_per_class),
                    knn_k=int(self.config.bezier_knn_k),
                    control_grid_size=4,
                    center_sampling=self.config.bezier_center_sampling,
                    seed=int(self.config.bezier_seed),
                    device="cpu",
                )
                patches_by_label = generate_bezier_patches_from_labeled_points(seed_xyz_obj, seed_labels_obj, patch_cfg)

                Nu = int(self.config.bezier_num_u)
                Nv = int(self.config.bezier_num_v)
                rho = float(self.config.bezier_rho)
                alpha = float(self.config.bezier_alpha)
                if Nu < 2 or Nv < 2:
                    raise ValueError("bezier_num_u and bezier_num_v must be >= 2.")
                if rho <= 0.0:
                    raise ValueError("bezier_rho must be > 0.")

                means_list: List[torch.Tensor] = []
                scales_list: List[torch.Tensor] = []
                quats_list: List[torch.Tensor] = []
                colors_list: List[torch.Tensor] = []
                sem_list: List[torch.Tensor] = []

                for lab, plist in patches_by_label.items():
                    # Patch-level mean color from points in this class (clean + stable).
                    lab_mask = seed_labels_obj == int(lab)
                    if bool(lab_mask.any()) and seed_rgb_obj.numel() > 0 and int(seed_rgb_obj.shape[0]) == int(seed_labels_obj.shape[0]):
                        col = seed_rgb_obj[lab_mask].float().mean(dim=0)  # (3,) in 0..255
                    else:
                        col = torch.tensor([127.0, 127.0, 127.0], dtype=torch.float32)

                    for patch in plist:
                        mode = str(getattr(self.config, "bezier_surface_mode", "open"))
                        if mode not in ("open", "closed", "both"):
                            raise ValueError(f"Unknown bezier_surface_mode={mode!r}. Expected 'open', 'closed', or 'both'.")

                        if mode == "open":
                            # ── Open mode ──────────────────────────────────────────
                            # Single Bezier surface with trainable, initially-planar
                            # control points.  At every training iteration the
                            # Gaussians are re-sampled from the (updated) patch, so
                            # the rendering loss back-propagates to the CPs.
                            X = patch.sample(num_u=Nu, num_v=Nv).float().unsqueeze(0)  # (1, Nu, Nv, 3)

                            pid = len(open_cp_list)
                            open_cp_list.append(patch.control_points.detach().float())
                            n_gauss_patch = Nu * Nv
                            open_patch_idx_list.append(
                                torch.full((n_gauss_patch,), pid, dtype=torch.long)
                            )
                            open_u_idx_list.append(
                                torch.arange(Nu, dtype=torch.long)
                                .unsqueeze(1).expand(Nu, Nv).reshape(-1)
                            )
                            open_v_idx_list.append(
                                torch.arange(Nv, dtype=torch.long)
                                .unsqueeze(0).expand(Nu, Nv).reshape(-1)
                            )

                            if bool(getattr(self.config, "bezier_surface_pruning_enabled", False)):
                                prune_surface_idx_list.append(
                                    torch.full((n_gauss_patch,), pid, dtype=torch.long)
                                )

                            extra_wall_surfaces = []

                        else:
                            # ── Closed / both modes ────────────────────────────────
                            # Construct a paired shell (out + in surfaces).
                            R_layers = int(self.config.bezier_num_r)
                            thickness = float(self.config.bezier_closed_thickness)
                            if R_layers < 2:
                                raise ValueError(
                                    "bezier_num_r must be >= 2 for closed/both mode."
                                )

                            X_out = patch.sample(num_u=Nu, num_v=Nv).float()
                            Xu_f0 = torch.zeros_like(X_out)
                            Xu_f0[1:-1] = X_out[2:] - X_out[:-2]
                            Xu_f0[0] = X_out[1] - X_out[0]
                            Xu_f0[-1] = X_out[-1] - X_out[-2]
                            tu0 = _safe_normalize(Xu_f0)

                            Xv_f0 = torch.zeros_like(X_out)
                            Xv_f0[:, 1:-1] = X_out[:, 2:] - X_out[:, :-2]
                            Xv_f0[:, 0] = X_out[:, 1] - X_out[:, 0]
                            Xv_f0[:, -1] = X_out[:, -1] - X_out[:, -2]
                            tv0 = _safe_normalize(Xv_f0)

                            n0 = _safe_normalize(torch.cross(tu0, tv0, dim=-1))
                            X_in_tgt = X_out - thickness * n0
                            cp_out = patch.control_points
                            deg_u = int(cp_out.shape[0]) - 1
                            deg_v = int(cp_out.shape[1]) - 1
                            cp_in = fit_bezier_control_points_from_grid(
                                X_in_tgt.to(dtype=cp_out.dtype), m=deg_u, n=deg_v,
                            ).to(device=cp_out.device, dtype=cp_out.dtype)

                            shell = BezierShellPatch(cp_out, cp_in)
                            X_layers = shell.sample_intermediate_layers(
                                num_r=R_layers, num_u=Nu, num_v=Nv
                            ).float()

                            keep_shell_params = (
                                bool(getattr(self.config, "bezier_topo_loss_enabled", False))
                                or bool(getattr(self.config, "bezier_attach_loss_enabled", False))
                                or bool(getattr(self.config, "bezier_reparam_enabled", False))
                                or bool(getattr(self.config, "bezier_surface_pruning_enabled", False))
                            )
                            shell_id = -1
                            if keep_shell_params:
                                shell_id = len(topo_cp_out_list)
                                topo_cp_out_list.append(cp_out.detach().to(dtype=torch.float32))
                                topo_cp_in_list.append(cp_in.detach().to(dtype=torch.float32))

                            include_walls = (
                                bool(getattr(self.config, "bezier_closed_include_walls", False))
                                and mode in ("closed", "both")
                            )
                            if include_walls:
                                walls = shell.sample_walls(num_r=R_layers, num_u=Nu, num_v=Nv)
                                extra_wall_surfaces = [w.unsqueeze(2) for w in walls.values()]
                            else:
                                extra_wall_surfaces = []

                            if mode == "closed":
                                X = X_layers[1:]
                                if X.shape[0] == 0:
                                    raise ValueError(
                                        "bezier_num_r too small: 'closed' mode requires R>=2."
                                    )
                            else:  # "both"
                                X = X_layers

                            attach_enabled = bool(getattr(self.config, "bezier_attach_loss_enabled", False))
                            reparam_enabled = bool(getattr(self.config, "bezier_reparam_enabled", False))
                            prune_enabled = bool(getattr(self.config, "bezier_surface_pruning_enabled", False))
                            if (attach_enabled or reparam_enabled or prune_enabled) and keep_shell_params and shell_id >= 0:
                                Rused = int(X.shape[0])
                                if mode == "closed":
                                    r_full = torch.arange(1, R_layers, dtype=torch.long)
                                else:
                                    r_full = torch.arange(0, R_layers, dtype=torch.long)
                                assert int(r_full.shape[0]) == Rused
                                rr = r_full.view(Rused, 1, 1).expand(Rused, Nu, Nv).reshape(-1)
                                uu = torch.arange(Nu, dtype=torch.long).view(1, Nu, 1).expand(Rused, Nu, Nv).reshape(-1)
                                vv = torch.arange(Nv, dtype=torch.long).view(1, 1, Nv).expand(Rused, Nu, Nv).reshape(-1)
                                ss = torch.full((rr.shape[0],), int(shell_id), dtype=torch.long)
                                if attach_enabled:
                                    attach_shell_idx_list.append(ss)
                                    attach_r_idx_list.append(rr)
                                    attach_u_idx_list.append(uu)
                                    attach_v_idx_list.append(vv)
                                if reparam_enabled:
                                    reparam_shell_idx_list.append(ss)
                                    reparam_r_idx_list.append(rr)
                                    reparam_u_idx_list.append(uu)
                                    reparam_v_idx_list.append(vv)
                                if prune_enabled:
                                    prune_surface_idx_list.append(ss)

                        # Tangent directions via finite differences (central; forward/backward at borders).
                        Xu_f = torch.zeros_like(X)
                        Xu_f[:, 1:-1] = X[:, 2:] - X[:, :-2]
                        Xu_f[:, 0] = X[:, 1] - X[:, 0]
                        Xu_f[:, -1] = X[:, -1] - X[:, -2]
                        tu = _safe_normalize(Xu_f)

                        Xv_f = torch.zeros_like(X)
                        Xv_f[:, :, 1:-1] = X[:, :, 2:] - X[:, :, :-2]
                        Xv_f[:, :, 0] = X[:, :, 1] - X[:, :, 0]
                        Xv_f[:, :, -1] = X[:, :, -1] - X[:, :, -2]
                        tv = _safe_normalize(Xv_f)

                        n = _safe_normalize(torch.cross(tu, tv, dim=-1))
                        # Re-orthonormalize tv to be orthogonal to tu and n.
                        tv = _safe_normalize(torch.cross(n, tu, dim=-1))

                        R_frame = torch.stack([tu, tv, n], dim=-1)  # (R,Nu,Nv,3,3) columns
                        # Use an already-implemented rotmat->quat conversion (COLMAP utils) and then reorder to xyzw
                        # to match `random_quat_tensor()` convention used by gsplat params in this codebase.
                        from nerfstudio.data.utils.colmap_parsing_utils import rotmat2qvec

                        R_np = R_frame.reshape(-1, 3, 3).detach().cpu().numpy()
                        q_wxyz = [rotmat2qvec(R_np[i]) for i in range(R_np.shape[0])]  # each is (4,) wxyz
                        import numpy as np

                        q_wxyz = torch.from_numpy(np.stack(q_wxyz).astype("float32"))
                        # wxyz -> xyzw
                        q_xyzw = torch.cat([q_wxyz[:, 1:4], q_wxyz[:, 0:1]], dim=-1)
                        q_xyzw = q_xyzw / torch.linalg.norm(q_xyzw, dim=-1, keepdim=True).clamp_min(1e-8)
                        quats_xyzw = q_xyzw.to(device=R_frame.device).reshape(X.shape[0], Nu, Nv, 4)

                        # Scales from adjacent sample distances (same scheme as open-surface init).
                        du = X[:, 1:] - X[:, :-1]  # (R,Nu-1,Nv,3)
                        dv = X[:, :, 1:] - X[:, :, :-1]  # (R,Nu,Nv-1,3)
                        sigma_u = torch.zeros((X.shape[0], Nu, Nv), dtype=torch.float32)
                        sigma_v = torch.zeros((X.shape[0], Nu, Nv), dtype=torch.float32)
                        sigma_u[:, :-1] = torch.linalg.norm(du, dim=-1) / rho
                        sigma_u[:, -1] = sigma_u[:, -2]
                        sigma_v[:, :, :-1] = torch.linalg.norm(dv, dim=-1) / rho
                        sigma_v[:, :, -1] = sigma_v[:, :, -2]
                        sigma_n = torch.full((X.shape[0], Nu, Nv), float(alpha / rho), dtype=torch.float32)

                        scales_lin = torch.stack([sigma_u, sigma_v, sigma_n], dim=-1).clamp_min(1e-6)
                        scales_log = torch.log(scales_lin)

                        Rcount = int(X.shape[0])
                        means_list.append(X.reshape(-1, 3))
                        scales_list.append(scales_log.reshape(-1, 3))
                        quats_list.append(quats_xyzw.reshape(-1, 4))
                        colors_list.append(col[None, :].repeat(Rcount * Nu * Nv, 1))
                        sem_list.append(torch.full((Rcount * Nu * Nv,), int(lab), dtype=torch.int64))

                        # Optionally add side walls (each treated as a thin surface, like open init).
                        for Xw in extra_wall_surfaces:
                            # Xw: (R, edge, 1, 3) where "u" dimension is edge samples and "v" has size 1.
                            Rw, Uw, Vw = int(Xw.shape[0]), int(Xw.shape[1]), int(Xw.shape[2])
                            assert Vw == 1
                            # Tangents: along edge direction only; create a dummy second tangent axis.
                            duw = Xw[:, 1:] - Xw[:, :-1]  # (R,Uw-1,1,3)
                            sigma_u_w = torch.zeros((Rw, Uw, 1), dtype=torch.float32)
                            sigma_u_w[:, :-1] = torch.linalg.norm(duw, dim=-1) / rho
                            sigma_u_w[:, -1] = sigma_u_w[:, -2]
                            sigma_v_w = torch.full((Rw, Uw, 1), 1e-6, dtype=torch.float32)
                            sigma_n_w = torch.full((Rw, Uw, 1), float(alpha / rho), dtype=torch.float32)
                            scales_log_w = torch.log(torch.stack([sigma_u_w, sigma_v_w, sigma_n_w], dim=-1).clamp_min(1e-6))

                            means_list.append(Xw.reshape(-1, 3))
                            scales_list.append(scales_log_w.reshape(-1, 3))
                            # Walls: simplest stable choice is identity quats (aligned to world axes).
                            # (This path is optional and can be improved later by building a proper local frame.)
                            quats_list.append(
                                torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=Xw.device)[None, :].repeat(
                                    Rw * Uw * 1, 1
                                )
                            )
                            colors_list.append(col[None, :].repeat(Rw * Uw * 1, 1))
                            sem_list.append(torch.full((Rw * Uw * 1,), int(lab), dtype=torch.int64))

                if len(means_list) > 0:
                    means0 = torch.cat(means_list, dim=0)
                    scales0 = torch.cat(scales_list, dim=0)
                    quats0 = torch.cat(quats_list, dim=0)
                    colors0 = torch.cat(colors_list, dim=0)  # (N,3) in 0..255
                    sem0 = torch.cat(sem_list, dim=0)

                    means = torch.nn.Parameter(means0)
                    scales = torch.nn.Parameter(scales0)
                    quats = torch.nn.Parameter(quats0)
                    num_points = int(means0.shape[0])

                    dim_sh = num_sh_bases(self.config.sh_degree)
                    shs = torch.zeros((num_points, dim_sh, 3), dtype=torch.float32)
                    if self.config.sh_degree > 0:
                        shs[:, 0, :3] = RGB2SH(colors0 / 255.0)
                        shs[:, 1:, 3:] = 0.0
                        features_dc = torch.nn.Parameter(shs[:, 0, :])
                        features_rest = torch.nn.Parameter(shs[:, 1:, :])
                    else:
                        features_dc = torch.nn.Parameter(torch.logit(colors0 / 255.0, eps=1e-10))
                        features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

                    # Store semantic label per Gaussian for later use (buffer).
                    self.register_buffer("gaussian_semantic_labels", sem0, persistent=True)
                    bezier_init_done = True
                else:
                    CONSOLE.log("[yellow]Bezier init enabled, but no patches were generated; falling back.[/yellow]")
            except Exception as e:
                CONSOLE.log(f"[yellow]Bezier init failed, falling back to standard seed init: {e}[/yellow]")

        # ── Open-mode: register trainable Bezier control points and index buffers ──
        if len(open_cp_list) > 0:
            try:
                self.bezier_open_cp = torch.nn.Parameter(
                    torch.stack(open_cp_list, dim=0)  # (S, G, G, 3)
                )
                self.register_buffer(
                    "bezier_open_patch_idx",
                    torch.cat(open_patch_idx_list, dim=0).to(dtype=torch.long),
                    persistent=True,
                )
                self.register_buffer(
                    "bezier_open_u_idx",
                    torch.cat(open_u_idx_list, dim=0).to(dtype=torch.long),
                    persistent=True,
                )
                self.register_buffer(
                    "bezier_open_v_idx",
                    torch.cat(open_v_idx_list, dim=0).to(dtype=torch.long),
                    persistent=True,
                )
                S_open = int(self.bezier_open_cp.shape[0])
                T_tex = int(getattr(self.config, "bezier_texture_size", 8))
                self.patch_textures = torch.nn.Parameter(
                    torch.rand(S_open, T_tex, T_tex, 4, dtype=torch.float32) * 0.5
                )
                if bool(getattr(self.config, "bezier_surface_pruning_enabled", False)):
                    self.register_buffer(
                        "bezier_surface_active",
                        torch.ones((S_open,), dtype=torch.bool),
                        persistent=True,
                    )
                    if len(prune_surface_idx_list) > 0:
                        self.register_buffer(
                            "bezier_surface_gaussian_surface_idx",
                            torch.cat(prune_surface_idx_list, dim=0).to(dtype=torch.long),
                            persistent=True,
                        )
                CONSOLE.log(
                    f"[green]Bezier open mode: {S_open} patches, "
                    f"{int(self.bezier_open_patch_idx.shape[0])} Gaussians[/green]"
                )
            except Exception as e:
                CONSOLE.log(f"[yellow]Warning: failed to register open-mode Bezier params: {e}[/yellow]")

        # Register trainable Bezier shell control points for closed/both topology regularization.
        if (
            (bool(getattr(self.config, "bezier_topo_loss_enabled", False)) or bool(getattr(self.config, "bezier_reparam_enabled", False)))
            and len(topo_cp_out_list) > 0
        ):
            try:
                self.bezier_shell_cp_out = torch.nn.Parameter(torch.stack(topo_cp_out_list, dim=0))  # (S,4,4,3)
                self.bezier_shell_cp_in = torch.nn.Parameter(torch.stack(topo_cp_in_list, dim=0))  # (S,4,4,3)
            except Exception as e:
                CONSOLE.log(f"[yellow]Warning: failed to register Bezier topology params (skipping topo loss): {e}[/yellow]")

        # Optional: attachment indices for Gaussians initialized from closed shells.
        if bool(getattr(self.config, "bezier_attach_loss_enabled", False)) and len(attach_shell_idx_list) > 0:
            try:
                self.register_buffer(
                    "bezier_attach_shell_idx", torch.cat(attach_shell_idx_list, dim=0).to(dtype=torch.long), persistent=True
                )
                self.register_buffer(
                    "bezier_attach_r_idx", torch.cat(attach_r_idx_list, dim=0).to(dtype=torch.long), persistent=True
                )
                self.register_buffer(
                    "bezier_attach_u_idx", torch.cat(attach_u_idx_list, dim=0).to(dtype=torch.long), persistent=True
                )
                self.register_buffer(
                    "bezier_attach_v_idx", torch.cat(attach_v_idx_list, dim=0).to(dtype=torch.long), persistent=True
                )
                # Used to guard against densification/pruning changing the Gaussian set.
                self._bezier_attach_num_points_init = int(torch.cat(attach_shell_idx_list, dim=0).shape[0])
            except Exception as e:
                CONSOLE.log(f"[yellow]Warning: failed to register Bezier attachment indices (skipping attach loss): {e}[/yellow]")

        # Optional: reparameterization indices (always used when enabled).
        if bool(getattr(self.config, "bezier_reparam_enabled", False)) and len(reparam_shell_idx_list) > 0:
            try:
                self.register_buffer(
                    "bezier_reparam_shell_idx",
                    torch.cat(reparam_shell_idx_list, dim=0).to(dtype=torch.long),
                    persistent=True,
                )
                self.register_buffer(
                    "bezier_reparam_r_idx",
                    torch.cat(reparam_r_idx_list, dim=0).to(dtype=torch.long),
                    persistent=True,
                )
                self.register_buffer(
                    "bezier_reparam_u_idx",
                    torch.cat(reparam_u_idx_list, dim=0).to(dtype=torch.long),
                    persistent=True,
                )
                self.register_buffer(
                    "bezier_reparam_v_idx",
                    torch.cat(reparam_v_idx_list, dim=0).to(dtype=torch.long),
                    persistent=True,
                )
            except Exception as e:
                CONSOLE.log(f"[yellow]Warning: failed to register Bezier reparam indices (disabling reparam): {e}[/yellow]")
                self.config.bezier_reparam_enabled = False

        # Optional: surface pruning mapping + active mask.
        if bool(getattr(self.config, "bezier_surface_pruning_enabled", False)) and len(prune_surface_idx_list) > 0:
            try:
                self.register_buffer(
                    "bezier_surface_gaussian_surface_idx",
                    torch.cat(prune_surface_idx_list, dim=0).to(dtype=torch.long),
                    persistent=True,
                )
                if hasattr(self, "bezier_shell_cp_out"):
                    S = int(self.bezier_shell_cp_out.shape[0])
                    self.register_buffer("bezier_surface_active", torch.ones((S,), dtype=torch.bool), persistent=True)
            except Exception as e:
                CONSOLE.log(f"[yellow]Warning: failed to register Bezier pruning buffers (disabling pruning): {e}[/yellow]")
                self.config.bezier_surface_pruning_enabled = False

        # Surface active mask for pruning (1 per shell patch).
        # (May have already been initialized above when registering pruning buffers.)
        if (
            bool(getattr(self.config, "bezier_surface_pruning_enabled", False))
            and hasattr(self, "bezier_shell_cp_out")
            and (not hasattr(self, "bezier_surface_active"))
        ):
            try:
                S = int(self.bezier_shell_cp_out.shape[0])
                self.register_buffer("bezier_surface_active", torch.ones((S,), dtype=torch.bool), persistent=True)
            except Exception as e:
                CONSOLE.log(f"[yellow]Warning: failed to init bezier_surface_active mask (disabling pruning): {e}[/yellow]")
                self.config.bezier_surface_pruning_enabled = False

        if not bezier_init_done:
            _sp = self.seed_points  # may be None; local copy we can filter

            # ── Optional: keep only object seeds (label > 0) for the classic path ──
            if (
                self.config.filter_seeds_by_segmentation
                and _sp is not None
                and not self.config.random_init
                and seed_sem is not None
            ):
                _labels = torch.as_tensor(seed_sem).detach().cpu().to(torch.int64)
                if int(_labels.shape[0]) == int(_sp[0].shape[0]):
                    _obj = _labels > 0
                    if bool(_obj.any()):
                        _sp0 = _sp[0][_obj]
                        _sp1 = _sp[1][_obj] if int(_sp[1].shape[0]) == int(_sp[0].shape[0]) else _sp[1]
                        _sp = (_sp0, _sp1)
                        CONSOLE.log(
                            f"[green]Classic init: keeping {int(_obj.sum())}/{int(_labels.shape[0])} "
                            f"object seeds (filtered by segmentation).[/green]"
                        )
                    else:
                        CONSOLE.log("[yellow]filter_seeds_by_segmentation: all labels are 0, keeping all seeds.[/yellow]")
                else:
                    CONSOLE.log(
                        f"[yellow]filter_seeds_by_segmentation: label count mismatch "
                        f"({int(_labels.shape[0])} vs {int(_sp[0].shape[0])}), skipping filter.[/yellow]"
                    )

            if _sp is not None and not self.config.random_init:
                means_param = torch.nn.Parameter(_sp[0])
            else:
                means_param = torch.nn.Parameter(
                    (torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale
                )
            means = means_param
            n_pts = int(means_param.shape[0])
            knn_k = min(3, max(1, n_pts - 1))
            if knn_k >= 1:
                distances, _ = k_nearest_sklearn(means_param.data, knn_k)
                avg_dist = distances.mean(dim=-1, keepdim=True)
                avg_dist = avg_dist.clamp(min=1e-6)
            else:
                avg_dist = torch.full((n_pts, 1), 0.01)
            scales_param = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
            scales = scales_param
            num_points = int(means_param.shape[0])
            quats_param = torch.nn.Parameter(random_quat_tensor(num_points))
            quats = quats_param
            dim_sh = num_sh_bases(self.config.sh_degree)

            if (
                _sp is not None
                and not self.config.random_init
                and _sp[1].shape[0] > 0
            ):
                shs = torch.zeros((_sp[1].shape[0], dim_sh, 3)).float().cuda()
                if self.config.sh_degree > 0:
                    shs[:, 0, :3] = RGB2SH(_sp[1] / 255)
                    shs[:, 1:, 3:] = 0.0
                else:
                    CONSOLE.log("use color only optimization with sigmoid activation")
                    shs[:, 0, :3] = torch.logit(_sp[1] / 255, eps=1e-10)
                features_dc = torch.nn.Parameter(shs[:, 0, :])
                features_rest = torch.nn.Parameter(shs[:, 1:, :])
            else:
                features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
                features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        # At this point, one of the init paths must have produced the Gaussian params.
        assert means is not None and scales is not None and quats is not None and features_dc is not None and features_rest is not None
        num_points = int(means.shape[0])

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        # Cache for Depth Anything 3 depth computed from *training GT images* (full resolution).
        # Keyed by cam_idx (image index). Stored on CPU to avoid bloating GPU memory.
        self._da3_depth_cache_fullres: Dict[int, torch.Tensor] = {}

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        # Strategy for GS densification
        if self.config.strategy == "default":
            # Strategy for GS densification
            self.strategy = DefaultStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=True,
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        elif self.config.strategy == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=self.config.max_gs_num,
                noise_lr=self.config.noise_lr,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.cull_alpha_thresh,
                verbose=False,
            )
            self.strategy_state = self.strategy.initialize_state()
        else:
            raise ValueError(f"""Splatfacto does not support strategy {self.config.strategy}
                             Currently, the supported strategies include default and mcmc.""")

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.config.sh_degree > 0:
            return self.features_dc
        else:
            return RGB2SH(torch.sigmoid(self.features_dc))

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        # Open-mode Bezier: means/scales come from the patch, skip standard
        # densification but still allow surface-level pruning.
        if hasattr(self, "bezier_open_cp"):
            if bool(getattr(self.config, "bezier_surface_pruning_enabled", False)):
                self._maybe_prune_bezier_surfaces()
            return
        # Shell reparameterization / surface pruning: also skip standard densification.
        if bool(getattr(self.config, "bezier_reparam_enabled", False)) or bool(getattr(self.config, "bezier_surface_pruning_enabled", False)):
            if bool(getattr(self.config, "bezier_surface_pruning_enabled", False)):
                self._maybe_prune_bezier_surfaces()
            return
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                packed=False,
            )
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=self.info,
                lr=self.schedulers["means"].get_last_lr()[0],  # the learning rate for the "means" attribute of the GS
            )
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            )
        )
        if self.config.export_end_of_training_outputs:
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN],
                    self.export_end_of_training_outputs,
                    args=[training_callback_attributes],
                )
            )
        if self.config.export_bezier_meshes:
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN],
                    self._export_bezier_meshes,
                    args=[training_callback_attributes],
                )
            )
        if self.config.export_gaussian_mesh:
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN],
                    self._export_gaussian_mesh,
                    args=[training_callback_attributes],
                )
            )
        return cbs

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers

    @torch.no_grad()
    def export_end_of_training_outputs(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
        """Export per-training-camera outputs to disk at the end of training.

        Writes folders (one file per cam_idx):
        - rgb: rendered RGB from the final model
        - depth_rasterizer: gsplat rasterizer expected depth (RGB+ED)
        - depth_da3: Depth Anything 3 metric depth inferred from the rendered RGB
        - depth_ellipsoid: geometric ray–ellipsoid first-hit depth from the final Gaussians
        
        """
        pipeline = training_callback_attributes.pipeline
        trainer = training_callback_attributes.trainer
        if pipeline is None or getattr(pipeline, "datamanager", None) is None:
            CONSOLE.log("[yellow]End-of-training export skipped: no pipeline/datamanager available.[/yellow]")
            return
        train_dataset = getattr(pipeline.datamanager, "train_dataset", None)
        if train_dataset is None or getattr(train_dataset, "cameras", None) is None:
            CONSOLE.log("[yellow]End-of-training export skipped: no train_dataset/cameras available.[/yellow]")
            return

        # Prefer writing into the experiment base dir (same place as config/checkpoints).
        base_dir: Path
        if trainer is not None and getattr(trainer, "config", None) is not None:
            try:
                base_dir = Path(trainer.config.get_base_dir())
            except Exception:
                base_dir = Path(".")
        else:
            base_dir = Path(".")

        out_root = base_dir / self.config.export_end_of_training_dirname
        rgb_dir = out_root / "rgb"
        rast_dir = out_root / "depth_rasterizer"
        da3_dir = out_root / "depth_da3"
        ell_dir = out_root / "depth_ellipsoid"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        rast_dir.mkdir(parents=True, exist_ok=True)
        da3_dir.mkdir(parents=True, exist_ok=True)
        ell_dir.mkdir(parents=True, exist_ok=True)

        try:
            import imageio.v3 as iio
        except Exception as e:  # pragma: no cover
            raise RuntimeError("imageio is required to export end-of-training images.") from e

        def _save_rgb_png(path: Path, rgb: torch.Tensor) -> None:
            # rgb: [H,W,3] float in [0,1]
            rgb_u8 = torch.clamp(rgb, 0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
            iio.imwrite(path, rgb_u8, extension=".png")

        def _save_depth_mm_png(path: Path, depth_m: torch.Tensor) -> None:
            # depth_m: [H,W,1] float meters -> uint16 millimeters (clipped)
            d = depth_m.squeeze(-1)
            d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            d_mm = torch.clamp(d * 1000.0, 0.0, 65535.0).to(torch.uint16).cpu().numpy()
            iio.imwrite(path, d_mm, extension=".png")

        num_cams = len(train_dataset)
        k = int(self.config.export_end_of_training_num_cameras)
        k = max(0, min(k, num_cams))
        if k == 0:
            CONSOLE.log("[yellow]End-of-training export skipped: export_end_of_training_num_cameras is 0.[/yellow]")
            return

        import random

        selected_cam_idxs = sorted(random.sample(range(num_cams), k=k))
        CONSOLE.log(
            f"[green]Exporting end-of-training outputs for {k}/{num_cams} random cameras to {out_root}[/green]"
        )

        # Render and export.
        was_training = self.training
        self.eval()

        try:
            from nerfstudio.models.da3_depth import get_da3_metric_estimator
        except Exception:
            get_da3_metric_estimator = None  # type: ignore
        try:
            from nerfstudio.models.ellipsoid_depth import EllipsoidDepthConfig, compute_ellipsoid_depth
        except Exception:
            EllipsoidDepthConfig = None  # type: ignore
            compute_ellipsoid_depth = None  # type: ignore

        for cam_idx in selected_cam_idxs:
            cam = train_dataset.cameras[cam_idx : cam_idx + 1].to(self.device)
            if cam.metadata is None:
                cam.metadata = {}
            cam.metadata["cam_idx"] = cam_idx

            outs = self.get_outputs(cam)
            rgb_out = outs.get("rgb", None)
            if not isinstance(rgb_out, torch.Tensor):
                CONSOLE.log(f"[yellow]RGB export skipped for cam {cam_idx}: outputs['rgb'] is not a Tensor.[/yellow]")
                continue
            rgb = rgb_out.detach().cpu()  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
            _save_rgb_png(rgb_dir / f"{cam_idx:05d}.png", rgb)

            # Rasterizer expected depth (from gsplat RGB+ED render_mode in eval).
            depth_rast_out = outs.get("depth", None)
            if isinstance(depth_rast_out, torch.Tensor):
                depth_rast = depth_rast_out.detach().cpu()
                if depth_rast.ndim == 2:
                    depth_rast = depth_rast[:, :, None]
                _save_depth_mm_png(rast_dir / f"{cam_idx:05d}.png", depth_rast)
            else:
                CONSOLE.log(
                    f"[yellow]Rasterizer depth export skipped for cam {cam_idx}: outputs['depth'] is not a Tensor.[/yellow]"
                )

            # DA3 depth from rendered RGB.
            if get_da3_metric_estimator is not None:
                try:
                    est = get_da3_metric_estimator(
                        model_id=self.config.da3_model_id.lower(),
                        max_side=self.config.da3_max_side,
                        use_half=self.config.da3_use_half,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    )
                    K0 = cam.get_intrinsics_matrices()[0].to(self.device)
                    focal = float(((K0[0, 0] + K0[1, 1]) * 0.5).item())
                    depth_da3 = est.infer_metric_depth(rgb.to(self.device), focal_px=focal)  # [H,W,1]
                    _save_depth_mm_png(da3_dir / f"{cam_idx:05d}.png", depth_da3)
                except Exception as e:
                    CONSOLE.log(f"[yellow]DA3 depth export failed for cam {cam_idx}: {e}[/yellow]")

            # Ellipsoid depth from final Gaussians.
            if compute_ellipsoid_depth is not None and EllipsoidDepthConfig is not None:
                try:
                    depth_cfg = EllipsoidDepthConfig(
                        method=self.config.ellipsoid_depth_method,
                        k=self.config.ellipsoid_depth_k,
                        tile_size=self.config.ellipsoid_depth_tile_size,
                        max_gaussians_per_tile=self.config.ellipsoid_depth_max_gaussians_per_tile,
                        output_depth_space=self.config.ellipsoid_depth_output_space,
                        ray_chunk_size=8192,
                        gauss_chunk_size=self.config.ellipsoid_depth_gauss_chunk_size,
                        debug=False,
                    )
                    alpha_mask = outs.get("accumulation", None)
                    if not isinstance(alpha_mask, torch.Tensor):
                        alpha_mask = None
                    depth_ell = compute_ellipsoid_depth(
                        camera=cam,
                        means=self.means,
                        scales=torch.exp(self.scales),
                        quats=self.quats,
                        alpha_mask=alpha_mask,
                        config=depth_cfg,
                        gsplat_meta=self.info if self.config.ellipsoid_depth_method == "tile" else None,
                    )
                    _save_depth_mm_png(ell_dir / f"{cam_idx:05d}.png", depth_ell)
                except Exception as e:
                    CONSOLE.log(f"[yellow]Ellipsoid depth export failed for cam {cam_idx}: {e}[/yellow]")

        if was_training:
            self.train()

    @torch.no_grad()
    def _export_bezier_meshes(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
        """Export Bezier surface patches as triangle-mesh PLY files at end of training."""
        has_open = hasattr(self, "bezier_open_cp")
        has_shell = hasattr(self, "bezier_shell_cp_out") and hasattr(self, "bezier_shell_cp_in")
        if not (has_open or has_shell):
            CONSOLE.log("[yellow]Bezier mesh export skipped: no Bezier control points found.[/yellow]")
            return

        trainer = training_callback_attributes.trainer
        if trainer is not None and getattr(trainer, "config", None) is not None:
            try:
                base_dir = Path(trainer.config.get_base_dir())
            except Exception:
                base_dir = Path(".")
        else:
            base_dir = Path(".")

        out_dir = base_dir / self.config.export_end_of_training_dirname / "bezier_meshes"
        Nu = int(self.config.bezier_num_u)
        Nv = int(self.config.bezier_num_v)

        try:
            if has_open:
                paths = export_bezier_patches_as_ply(
                    self.bezier_open_cp.detach(),
                    out_dir,
                    num_u=Nu,
                    num_v=Nv,
                    prefix="bezier_open",
                )
                CONSOLE.log(f"[green]Exported {len(paths)} open-mode Bezier mesh file(s) to {out_dir}[/green]")

            if has_shell:
                paths = export_bezier_patches_as_ply(
                    self.bezier_shell_cp_out.detach(),
                    out_dir,
                    num_u=Nu,
                    num_v=Nv,
                    control_points_in=self.bezier_shell_cp_in.detach(),
                    num_r=int(self.config.bezier_num_r),
                    prefix="bezier_shell",
                )
                CONSOLE.log(f"[green]Exported {len(paths)} shell-mode Bezier mesh file(s) to {out_dir}[/green]")
        except Exception as e:
            CONSOLE.log(f"[yellow]Bezier mesh export failed: {e}[/yellow]")

    @torch.no_grad()
    def _export_gaussian_mesh(self, training_callback_attributes: TrainingCallbackAttributes, step: int):
        """Extract a triangle mesh from the Gaussian field and export as PLY at end of training.

        Replicates the same means / scales / opacities / quats resolution logic
        used by ``get_outputs`` (open-mode Bezier resample, shell reparam, surface
        pruning) so that the exported mesh is faithful to the final rendering.
        """
        trainer = training_callback_attributes.trainer
        if trainer is not None and getattr(trainer, "config", None) is not None:
            try:
                base_dir = Path(trainer.config.get_base_dir())
            except Exception:
                base_dir = Path(".")
        else:
            base_dir = Path(".")

        out_dir = base_dir / self.config.export_end_of_training_dirname / "gaussian_mesh"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Resolve current means / scales (same logic as get_outputs) ──
        opacities_all = self.opacities
        quats_all = self.quats

        if hasattr(self, "bezier_open_cp"):
            means_all, scales_log_all, open_keep = self._bezier_open_resample_means_scales()
            if open_keep is not None:
                opacities_all = opacities_all[open_keep]
                quats_all = quats_all[open_keep]
        elif (
            bool(getattr(self.config, "bezier_reparam_enabled", False))
            and hasattr(self, "bezier_shell_cp_out")
        ):
            means_all, scales_log_all = self._bezier_reparam_means_scales()
            if (
                bool(getattr(self.config, "bezier_surface_pruning_enabled", False))
                and hasattr(self, "bezier_surface_active")
                and hasattr(self, "bezier_reparam_shell_idx")
            ):
                keep = self.bezier_surface_active[self.bezier_reparam_shell_idx]
                opacities_all = opacities_all[keep]
                quats_all = quats_all[keep]
        else:
            means_all = self.means
            scales_log_all = self.scales
            # Shell-mode surface pruning WITHOUT reparam
            if (
                bool(getattr(self.config, "bezier_surface_pruning_enabled", False))
                and hasattr(self, "bezier_surface_active")
            ):
                ss_map = None
                for attr in (
                    "bezier_surface_gaussian_surface_idx",
                    "bezier_reparam_shell_idx",
                    "bezier_attach_shell_idx",
                ):
                    if hasattr(self, attr):
                        ss_map = getattr(self, attr)
                        break
                if ss_map is not None and int(ss_map.shape[0]) == int(means_all.shape[0]):
                    keep = self.bezier_surface_active[ss_map]
                    means_all = means_all[keep]
                    scales_log_all = scales_log_all[keep]
                    opacities_all = opacities_all[keep]
                    quats_all = quats_all[keep]

        scales_exp = torch.exp(scales_log_all)
        opacities_sig = torch.sigmoid(opacities_all).squeeze(-1)

        if means_all.shape[0] == 0:
            CONSOLE.log("[yellow]Gaussian mesh export skipped: no active Gaussians.[/yellow]")
            return

        CONSOLE.log(
            f"[green]Extracting Gaussian mesh ({self.config.gaussian_mesh_resolution}³, "
            f"τ={self.config.gaussian_mesh_isovalue}, {means_all.shape[0]} Gaussians) …[/green]"
        )

        try:
            path = export_gaussian_mesh_as_ply(
                means_all.detach(),
                scales_exp.detach(),
                quats_all.detach(),
                opacities_sig.detach(),
                out_dir / "gaussian_mesh.ply",
                resolution=int(self.config.gaussian_mesh_resolution),
                isovalue=float(self.config.gaussian_mesh_isovalue),
                culling_sigma=float(self.config.gaussian_mesh_culling_sigma),
                chunk_size=int(self.config.gaussian_mesh_chunk_size),
            )
            if path is not None:
                CONSOLE.log(f"[green]Gaussian mesh exported to {path}[/green]")
            else:
                CONSOLE.log("[yellow]Gaussian mesh extraction returned no result.[/yellow]")
        except Exception as e:
            CONSOLE.log(f"[yellow]Gaussian mesh export failed: {e}[/yellow]")

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        gps: Dict[str, List[Parameter]] = {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

        # Open-mode Bezier: control points replace means/scales; per-patch textures drive color/opacity.
        if hasattr(self, "bezier_open_cp"):
            gps["means"] = [self.bezier_open_cp]
            gps.pop("scales", None)
            if hasattr(self, "patch_textures"):
                gps["patch_textures"] = [self.patch_textures]
            return gps

        # Shell-mode: trainable shell CPs (topology / attach / reparam).
        if hasattr(self, "bezier_shell_cp_out") and (
            bool(getattr(self.config, "bezier_topo_loss_enabled", False))
            or bool(getattr(self.config, "bezier_attach_loss_enabled", False))
            or bool(getattr(self.config, "bezier_reparam_enabled", False))
        ):
            gps["means"] = list(gps.get("means", [])) + [self.bezier_shell_cp_out, self.bezier_shell_cp_in]

        if bool(getattr(self.config, "bezier_reparam_enabled", False)) and hasattr(self, "bezier_shell_cp_out"):
            gps["means"] = [self.bezier_shell_cp_out, self.bezier_shell_cp_in]
            gps.pop("scales", None)
        return gps

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def _apply_bilateral_grid(self, rgb: torch.Tensor, cam_idx: int, H: int, W: int) -> torch.Tensor:
        # make xy grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H, device=self.device),
            torch.linspace(0, 1.0, W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        out = slice(
            bil_grids=self.bil_grids,
            rgb=rgb,
            xy=grid_xy,
            grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
        )
        return out["rgb"]

    @staticmethod
    def _linear_schedule(step: int, start: float, end: float, max_steps: int) -> float:
        if max_steps <= 0:
            return float(end)
        t = float(max(0, min(step, max_steps))) / float(max_steps)
        return float(start) * (1.0 - t) + float(end) * t

    @torch.no_grad()
    def _maybe_prune_bezier_surfaces(self) -> None:
        """Adaptive pruning for Bezier surfaces (deactivate redundant/degenerate patches).

        Works for both open-mode (single surface per patch) and shell-mode
        (paired out/in surfaces).  Updates a boolean mask ``bezier_surface_active``
        so that downstream rendering excludes Gaussians of inactive patches.
        """
        if not bool(getattr(self.config, "bezier_surface_pruning_enabled", False)):
            return
        if not hasattr(self, "bezier_surface_active"):
            return
        is_open = hasattr(self, "bezier_open_cp")
        is_shell = hasattr(self, "bezier_shell_cp_out")
        if not (is_open or is_shell):
            return
        if int(self.step) < int(getattr(self.config, "bezier_prune_start_step", 0)):
            return
        every = int(getattr(self.config, "bezier_prune_every", 0))
        if every <= 0 or (int(self.step) % every) != 0:
            return

        active = self.bezier_surface_active
        S = int(active.shape[0])
        if S == 0:
            return

        device = self.device

        # Schedules.
        tau_op = self._linear_schedule(
            int(self.step),
            float(getattr(self.config, "bezier_prune_tau_opacity_start", 0.01)),
            float(getattr(self.config, "bezier_prune_tau_opacity_end", 0.001)),
            int(getattr(self.config, "bezier_prune_tau_opacity_max_steps", 10000)),
        )
        tau_area = float(getattr(self.config, "bezier_prune_tau_area", 0.0))

        # Mapping from Gaussians → surface id.
        if hasattr(self, "bezier_surface_gaussian_surface_idx"):
            ss = self.bezier_surface_gaussian_surface_idx.to(device=device)
        elif is_open and hasattr(self, "bezier_open_patch_idx"):
            ss = self.bezier_open_patch_idx.to(device=device)
        elif hasattr(self, "bezier_reparam_shell_idx"):
            ss = self.bezier_reparam_shell_idx.to(device=device)
        elif hasattr(self, "bezier_attach_shell_idx"):
            ss = self.bezier_attach_shell_idx.to(device=device)
        else:
            return

        # Per-surface mean opacity from associated Gaussians.
        alpha_g = torch.sigmoid(self.opacities.detach().to(device=device)).squeeze(-1)  # (N,)
        if int(alpha_g.shape[0]) != int(ss.shape[0]):
            return
        sum_op = torch.zeros((S,), device=device, dtype=torch.float32)
        cnt = torch.zeros((S,), device=device, dtype=torch.float32)
        sum_op.scatter_add_(0, ss, alpha_g.to(dtype=torch.float32))
        cnt.scatter_add_(0, ss, torch.ones_like(alpha_g, dtype=torch.float32))
        mean_op = sum_op / cnt.clamp_min(1.0)

        # Geometric stats: area (and optionally thickness for shell mode).
        Nu = max(2, int(getattr(self.config, "bezier_prune_area_num_u", 20)))
        Nv = max(2, int(getattr(self.config, "bezier_prune_area_num_v", 20)))

        if is_open:
            dtype_cp = self.bezier_open_cp.dtype
            X_surf = sample_bezier_surfaces(
                self.bezier_open_cp.to(device=device, dtype=dtype_cp),
                num_u=Nu, num_v=Nv,
            )  # (S, Nu, Nv, 3)
            thick = None
        else:
            dtype_cp = self.bezier_shell_cp_out.dtype
            X2 = sample_paired_bezier_surfaces(
                self.bezier_shell_cp_out.to(device=device, dtype=dtype_cp),
                self.bezier_shell_cp_in.to(device=device, dtype=dtype_cp),
                num_r=2, num_u=Nu, num_v=Nv,
            )  # (S, 2, Nu, Nv, 3)
            X_surf = X2[:, 0]
            thick = torch.linalg.norm(X2[:, 0] - X2[:, 1], dim=-1).mean(dim=(1, 2))  # (S,)

        p00 = X_surf[:, :-1, :-1, :]
        p10 = X_surf[:, 1:, :-1, :]
        p01 = X_surf[:, :-1, 1:, :]
        area_cells = torch.linalg.norm(torch.cross(p10 - p00, p01 - p00, dim=-1), dim=-1)
        area = area_cells.sum(dim=(1, 2))  # (S,)

        to_remove = torch.zeros((S,), device=device, dtype=torch.bool)

        # Low-opacity pruning.
        to_remove |= (mean_op < float(tau_op))

        # Small-area pruning.
        if tau_area > 0.0:
            to_remove |= (area < float(tau_area))

        # Collapsed-shell pruning (only for shell mode).
        if thick is not None:
            tau_th = self._linear_schedule(
                int(self.step),
                float(getattr(self.config, "bezier_prune_tau_thick_start", 0.002)),
                float(getattr(self.config, "bezier_prune_tau_thick_end", 0.0005)),
                int(getattr(self.config, "bezier_prune_tau_thick_max_steps", 10000)),
            )
            if tau_th > 0.0:
                to_remove |= (thick < float(tau_th))

        to_remove &= active
        if bool(to_remove.any()):
            new_active = active.clone()
            new_active[to_remove] = False
            self.bezier_surface_active = new_active

    def _bezier_reparam_means_scales(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-Gaussian means and log-scales as functions of the Bezier shell control points.

        Returns:
            means: (N,3)
            scales_log: (N,3) in log-space (matching the rest of this file)
        """
        if not hasattr(self, "bezier_shell_cp_out"):
            raise RuntimeError("Bezier reparameterization requested but no bezier_shell_cp_out/in parameters exist.")
        if not hasattr(self, "bezier_reparam_shell_idx"):
            raise RuntimeError("Bezier reparameterization requested but reparam index buffers are missing.")

        cp_out = self.bezier_shell_cp_out
        cp_in = self.bezier_shell_cp_in
        Xs = sample_paired_bezier_surfaces(
            cp_out,
            cp_in,
            num_r=int(getattr(self.config, "bezier_num_r", 5)),
            num_u=int(getattr(self.config, "bezier_num_u", 20)),
            num_v=int(getattr(self.config, "bezier_num_v", 20)),
        )  # (S,R,U,V,3)

        ss = self.bezier_reparam_shell_idx
        rr = self.bezier_reparam_r_idx
        uu = self.bezier_reparam_u_idx
        vv = self.bezier_reparam_v_idx
        if bool(getattr(self.config, "bezier_surface_pruning_enabled", False)) and hasattr(self, "bezier_surface_active"):
            keep = self.bezier_surface_active[ss]
            ss = ss[keep]
            rr = rr[keep]
            uu = uu[keep]
            vv = vv[keep]
        means = Xs[ss, rr, uu, vv]  # (N,3) (N is after optional pruning mask)

        rho = float(getattr(self.config, "bezier_rho", 20.0))
        alpha = float(getattr(self.config, "bezier_alpha", 1.0))
        if rho <= 0.0:
            raise ValueError("bezier_rho must be > 0 for reparameterization.")

        # Tangential scales based on neighbor spacing on the sampled grid (same spirit as init code).
        du = Xs[:, :, 1:, :, :] - Xs[:, :, :-1, :, :]  # (S,R,U-1,V,3)
        dv = Xs[:, :, :, 1:, :] - Xs[:, :, :, :-1, :]  # (S,R,U,V-1,3)
        # IMPORTANT: avoid in-place updates that read from and then write into the same tensor
        # (can break autograd with "modified by an inplace operation" via AsStrided views).
        sigma_u_base = torch.linalg.norm(du, dim=-1) / rho  # (S,R,U-1,V)
        sigma_u_last = sigma_u_base[:, :, -1:, :].clamp_min(0.0)  # (S,R,1,V)
        sigma_u = torch.cat([sigma_u_base, sigma_u_last], dim=2)  # (S,R,U,V)

        sigma_v_base = torch.linalg.norm(dv, dim=-1) / rho  # (S,R,U,V-1)
        sigma_v_last = sigma_v_base[:, :, :, -1:].clamp_min(0.0)  # (S,R,U,1)
        sigma_v = torch.cat([sigma_v_base, sigma_v_last], dim=3)  # (S,R,U,V)
        sigma_n = torch.full_like(sigma_u, float(alpha / rho))

        su = sigma_u[ss, rr, uu, vv]
        sv = sigma_v[ss, rr, uu, vv]
        sn = sigma_n[ss, rr, uu, vv]
        scales_lin = torch.stack([su, sv, sn], dim=-1).clamp_min(1e-6)
        scales_log = torch.log(scales_lin)
        return means, scales_log

    def _bezier_open_resample_means_scales(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Re-sample Gaussian means and log-scales from the open-mode Bezier patches.

        The control points in ``self.bezier_open_cp`` are trainable nn.Parameters,
        so the returned tensors are part of the autograd graph and the rendering
        loss will back-propagate to the control points.

        Returns:
            means: (N', 3)
            scales_log: (N', 3) in log-space
            keep_mask: (N,) bool tensor if surface pruning is active (None otherwise).
                       When not None, only Gaussians where keep_mask is True are
                       returned; the caller must apply the same mask to the other
                       per-Gaussian arrays (opacities, quats, features, …).
        """
        if not hasattr(self, "bezier_open_cp"):
            raise RuntimeError("Open-mode Bezier resample requested but bezier_open_cp is missing.")

        cp = self.bezier_open_cp  # (S, G, G, 3)
        Nu = int(self.config.bezier_num_u)
        Nv = int(self.config.bezier_num_v)
        X = sample_bezier_surfaces(cp, num_u=Nu, num_v=Nv)  # (S, Nu, Nv, 3)

        pp = self.bezier_open_patch_idx
        uu = self.bezier_open_u_idx
        vv = self.bezier_open_v_idx

        # Surface pruning: filter out Gaussians belonging to inactive patches.
        keep_mask: Optional[torch.Tensor] = None
        if (
            bool(getattr(self.config, "bezier_surface_pruning_enabled", False))
            and hasattr(self, "bezier_surface_active")
        ):
            keep_mask = self.bezier_surface_active[pp]
            pp = pp[keep_mask]
            uu = uu[keep_mask]
            vv = vv[keep_mask]

        means = X[pp, uu, vv]  # (N', 3)

        rho = float(self.config.bezier_rho)
        alpha = float(self.config.bezier_alpha)
        if rho <= 0.0:
            raise ValueError("bezier_rho must be > 0 for open-mode reparameterization.")

        du = X[:, 1:, :, :] - X[:, :-1, :, :]  # (S, Nu-1, Nv, 3)
        dv = X[:, :, 1:, :] - X[:, :, :-1, :]  # (S, Nu, Nv-1, 3)

        sigma_u_base = torch.linalg.norm(du, dim=-1) / rho  # (S, Nu-1, Nv)
        sigma_u = torch.cat([sigma_u_base, sigma_u_base[:, -1:, :]], dim=1)  # (S, Nu, Nv)

        sigma_v_base = torch.linalg.norm(dv, dim=-1) / rho  # (S, Nu, Nv-1)
        sigma_v = torch.cat([sigma_v_base, sigma_v_base[:, :, -1:]], dim=2)  # (S, Nu, Nv)

        sigma_n = torch.full_like(sigma_u, float(alpha / rho))

        su = sigma_u[pp, uu, vv]
        sv = sigma_v[pp, uu, vv]
        sn = sigma_n[pp, uu, vv]
        scales_log = torch.log(torch.stack([su, sv, sn], dim=-1).clamp_min(1e-6))
        return means, scales_log, keep_mask

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a Cameras")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            means_all = self.means
            if hasattr(self, "bezier_open_cp"):
                means_all, _, _ = self._bezier_open_resample_means_scales()
            elif bool(getattr(self.config, "bezier_reparam_enabled", False)) and hasattr(self, "bezier_shell_cp_out"):
                means_all, _ = self._bezier_reparam_means_scales()
            crop_ids = self.crop_box.within(means_all).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        opacities_all = self.opacities
        features_dc_all = self.features_dc
        features_rest_all = self.features_rest
        quats_all = self.quats
        means_all = self.means
        scales_all = self.scales

        # Shell-mode surface pruning WITHOUT reparam: filter Gaussians of inactive surfaces.
        # (Open-mode pruning is handled inside _bezier_open_resample_means_scales below.)
        if (
            bool(getattr(self.config, "bezier_surface_pruning_enabled", False))
            and (not hasattr(self, "bezier_open_cp"))
            and (not bool(getattr(self.config, "bezier_reparam_enabled", False)))
            and hasattr(self, "bezier_surface_active")
        ):
            active_surf = self.bezier_surface_active
            ss_map = None
            if hasattr(self, "bezier_surface_gaussian_surface_idx"):
                ss_map = self.bezier_surface_gaussian_surface_idx
            elif hasattr(self, "bezier_reparam_shell_idx"):
                ss_map = self.bezier_reparam_shell_idx
            elif hasattr(self, "bezier_attach_shell_idx"):
                ss_map = self.bezier_attach_shell_idx
            if ss_map is not None and int(ss_map.shape[0]) == int(means_all.shape[0]):
                keep = active_surf[ss_map]
                opacities_all = opacities_all[keep]
                features_dc_all = features_dc_all[keep]
                features_rest_all = features_rest_all[keep]
                quats_all = quats_all[keep]
                means_all = means_all[keep]
                scales_all = scales_all[keep]
        # Open-mode Bezier reparameterization: resample means/scales from the
        # (updated) control points every forward pass.
        if hasattr(self, "bezier_open_cp"):
            means_all, scales_all, open_keep = self._bezier_open_resample_means_scales()
            if open_keep is not None:
                opacities_all = opacities_all[open_keep]
                features_dc_all = features_dc_all[open_keep]
                features_rest_all = features_rest_all[open_keep]
                quats_all = quats_all[open_keep]
            # Per-(u,v) color/opacity from patch textures: rgba = texture_i[u_idx, v_idx]
            if hasattr(self, "patch_textures"):
                pp = self.bezier_open_patch_idx
                uu = self.bezier_open_u_idx
                vv = self.bezier_open_v_idx
                if open_keep is not None:
                    pp = pp[open_keep]
                    uu = uu[open_keep]
                    vv = vv[open_keep]
                Nu = int(self.config.bezier_num_u)
                Nv = int(self.config.bezier_num_v)
                T_tex = int(self.config.bezier_texture_size)
                tu = (uu * (T_tex - 1) / max(1, Nu - 1)).long().clamp(0, T_tex - 1)
                tv = (vv * (T_tex - 1) / max(1, Nv - 1)).long().clamp(0, T_tex - 1)
                rgba = self.patch_textures[pp, tu, tv]  # (N, 4)
                rgba_01 = torch.sigmoid(rgba)
                dim_sh = num_sh_bases(self.config.sh_degree)
                if self.config.sh_degree > 0:
                    features_dc_all = RGB2SH(rgba_01[:, :3])
                    features_rest_all = torch.zeros(
                        pp.shape[0], dim_sh - 1, 3,
                        device=features_dc_all.device, dtype=features_dc_all.dtype,
                    )
                else:
                    features_dc_all = torch.logit(rgba_01[:, :3].clamp(1e-6, 1.0 - 1e-6), eps=1e-10)
                    features_rest_all = torch.zeros(
                        pp.shape[0], dim_sh - 1, 3,
                        device=features_dc_all.device, dtype=features_dc_all.dtype,
                    )
                opacities_all = torch.logit(
                    rgba_01[:, 3:4].clamp(1e-6, 1.0 - 1e-6), eps=1e-6
                )
        elif bool(getattr(self.config, "bezier_reparam_enabled", False)) and hasattr(self, "bezier_shell_cp_out"):
            means_all, scales_all = self._bezier_reparam_means_scales()
            if bool(getattr(self.config, "bezier_surface_pruning_enabled", False)) and hasattr(self, "bezier_surface_active"):
                keep = self.bezier_surface_active[self.bezier_reparam_shell_idx]
                opacities_all = opacities_all[keep]
                features_dc_all = features_dc_all[keep]
                features_rest_all = features_rest_all[keep]
                quats_all = quats_all[keep]

        if crop_ids is not None:
            opacities_crop = opacities_all[crop_ids]
            means_crop = means_all[crop_ids]
            features_dc_crop = features_dc_all[crop_ids]
            features_rest_crop = features_rest_all[crop_ids]
            scales_crop = scales_all[crop_ids]
            quats_crop = quats_all[crop_ids]
        else:
            opacities_crop = opacities_all
            means_crop = means_all
            features_dc_crop = features_dc_all
            features_rest_crop = features_rest_all
            scales_crop = scales_all
            quats_crop = quats_all

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        # Guard: if all Gaussians were filtered out (e.g., by surface pruning or crop box),
        # return an empty image immediately instead of calling rasterization with N=0.
        if means_crop.shape[0] == 0:
            return self.get_empty_outputs(
                int(camera.width.item()), int(camera.height.item()), self.background_color
            )

        camera_scale_fac = self._get_downscale_factor()
        # Full-res intrinsics (useful for depth-supervision targets derived from the GT image).
        try:
            K_full0 = camera.get_intrinsics_matrices()[0].to(self.device)
            focal_px_full = float(((K_full0[0, 0] + K_full0[1, 1]) * 0.5).item())
        except Exception:
            focal_px_full = 0.0
        cam_idx = -1
        if camera.metadata is not None and "cam_idx" in camera.metadata:
            try:
                cam_idx = int(camera.metadata["cam_idx"])
            except Exception:
                cam_idx = -1
        # IMPORTANT: never mutate the viewer's camera in-place. `rescale_output_resolution()` modifies
        # height/width with integer rounding, and repeated calls can drift to 0x0 and yield a black viewer.
        render_camera = dataclasses.replace(camera)
        render_camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = render_camera.get_intrinsics_matrices().cuda()
        W, H = int(render_camera.width.item()), int(render_camera.height.item())
        self.last_size = (H, W)

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training or (self.training and self.config.loss == "depth"):
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(  # type: ignore[reportPossiblyUnboundVariable]
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training:
            skip_densify = (
                hasattr(self, "bezier_open_cp")
                or bool(getattr(self.config, "bezier_reparam_enabled", False))
                or bool(getattr(self.config, "bezier_surface_pruning_enabled", False))
            )
            if not skip_densify:
                self.strategy.step_pre_backward(
                    self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
                )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if render_camera.metadata is not None and "cam_idx" in render_camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, render_camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        # Optional: geometric ellipsoid depth (ray–ellipsoid first hit).
        # IMPORTANT: this is intentionally disabled during training to avoid slowing training down
        # and to prevent GPU OOM from additional large temporary allocations.
        depth_ellipsoid = None
        if (not self.training) and self.config.depth_mode in ["ellipsoid", "both"]:
            try:
                from nerfstudio.models.ellipsoid_depth import EllipsoidDepthConfig, compute_ellipsoid_depth

                depth_cfg = EllipsoidDepthConfig(
                    method=self.config.ellipsoid_depth_method,
                    k=self.config.ellipsoid_depth_k,
                    tile_size=self.config.ellipsoid_depth_tile_size,
                    max_gaussians_per_tile=self.config.ellipsoid_depth_max_gaussians_per_tile,
                    output_depth_space=self.config.ellipsoid_depth_output_space,
                    ray_chunk_size=8192,
                    gauss_chunk_size=self.config.ellipsoid_depth_gauss_chunk_size,
                    debug=self.config.ellipsoid_depth_debug,
                )
                depth_ellipsoid = compute_ellipsoid_depth(
                    camera=render_camera,
                    means=means_crop,
                    scales=torch.exp(scales_crop),
                    quats=quats_crop,
                    alpha_mask=alpha.squeeze(0),
                    config=depth_cfg,
                    gsplat_meta=self.info if self.config.ellipsoid_depth_method == "tile" else None,
                )
            except Exception as e:
                CONSOLE.print_exception()   # oppure logger.exception(...)
                CONSOLE.log(f"[yellow]Ellipsoid depth failed, falling back to rasterizer depth: {e}[/yellow]")
                depth_ellipsoid = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        outputs: Dict[str, Union[torch.Tensor, List]] = {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore
        # Aux info for depth-supervision (safe to ignore for most losses).
        outputs["cam_idx"] = torch.tensor(cam_idx, device=self.device, dtype=torch.long)
        outputs["focal_px_full"] = torch.tensor(focal_px_full, device=self.device, dtype=torch.float32)

        # Optional: Depth Anything 3 metric depth from rendered RGB (viewer visualization).
        if (not self.training) and self.config.da3_depth_enabled:
            try:
                from nerfstudio.models.da3_depth import get_da3_metric_estimator

                est = get_da3_metric_estimator(
                    model_id=self.config.da3_model_id.lower(),
                    max_side=self.config.da3_max_side,
                    use_half=self.config.da3_use_half,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                )
                K0 = render_camera.get_intrinsics_matrices()[0].to(self.device)
                focal = float(((K0[0, 0] + K0[1, 1]) * 0.5).item())
                depth_da3 = est.infer_metric_depth(rgb.squeeze(0), focal_px=focal)
                outputs["depth_da3"] = depth_da3
            except Exception as e:
                CONSOLE.log(f"[yellow]DA3 depth failed, skipping: {e}[/yellow]")

        if depth_ellipsoid is not None:
            outputs["depth_ellipsoid"] = depth_ellipsoid
            if self.config.depth_mode == "ellipsoid":
                outputs["depth"] = depth_ellipsoid

        return outputs

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Optional segmentation mask: enable loss only on the object-of-interest pixels.
        # We apply the mask pixel-wise and normalize the reduction by the number of active pixels
        # (so the loss magnitude does not depend on object size).
        mask: Optional[torch.Tensor] = None
        gt_img_unmasked = gt_img
        pred_img_unmasked = pred_img
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask_t: torch.Tensor = self._downscale_if_required(batch["mask"]).to(self.device)
            if mask_t.ndim == 2:
                mask_t = mask_t[:, :, None]
            # Ensure float mask in [0,1] and shape [H,W,1]
            mask_t = mask_t.to(dtype=gt_img.dtype)
            mask_t = torch.clamp(mask_t, 0.0, 1.0)
            assert mask_t.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            assert mask_t.shape[-1] == 1
            # Debug: print mask statistics
            active_pixels = mask_t.sum().item()
            total_pixels = mask_t.numel()
            active_percent = 100.0 * active_pixels / total_pixels if total_pixels > 0 else 0.0
            CONSOLE.log(f"[cyan]Mask stats: {int(active_pixels)}/{int(total_pixels)} active pixels ({active_percent:.1f}%)[/cyan]")
            mask = mask_t

            # For SSIM (which is unweighted), we still multiply images pixel-wise so only masked pixels contribute.
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        def _masked_mean(x: torch.Tensor, mask_hw1: Optional[torch.Tensor]) -> torch.Tensor:
            """Compute mean over masked pixels (mask shape [H,W,1]); if mask is None, fall back to x.mean()."""
            if mask_hw1 is None:
                return x.mean()
            # Broadcast mask to match x channels if needed.
            m = mask_hw1
            if x.ndim == 3 and x.shape[-1] != 1:
                m = m.expand_as(x)
            denom = m.sum().clamp_min(1.0)
            return (x * m).sum() / denom
        
        simloss = torch.tensor(0.0).to(self.device)
        if self.config.loss == "depth":
            pred_depth = outputs.get("depth", None)
            if pred_depth is None:
                raise RuntimeError(
                    "config.loss='depth' requires expected depth from the renderer (RGB+ED), but outputs['depth'] is None."
                )
            # Use depth from batch if available (pre-computed during data processing)
            if "depth_image" in batch:
                gt_depth = batch["depth_image"].to(self.device)
                gt_depth = self._downscale_if_required(gt_depth)
            else:
                # Fallback: compute DA3 depth on-the-fly (slower, for backwards compatibility)
                cam_idx_t = outputs.get("cam_idx", None)
                cam_idx = int(cam_idx_t.item()) if torch.is_tensor(cam_idx_t) else -1
                focal_t = outputs.get("focal_px_full", None)
                focal_px = float(focal_t.item()) if torch.is_tensor(focal_t) else 0.0
                da3_depth_full = self._get_or_compute_da3_depth_fullres(cam_idx=cam_idx, image=batch["image"], focal_px=focal_px)
                gt_depth = self._downscale_if_required(da3_depth_full.to(self.device))
            if mask is not None:
                # pred_depth / gt_depth are [H,W,1] (or [H,W]); normalize reduction by active pixels.
                if pred_depth.ndim == 2:
                    pred_depth = pred_depth[:, :, None]
                if gt_depth.ndim == 2:
                    gt_depth = gt_depth[:, :, None]
                Ll1 = _masked_mean(torch.abs(gt_depth - pred_depth), mask)
            else:
                Ll1 = torch.abs(gt_depth - pred_depth).mean()
        else:
            if mask is not None:
                # Use masked mean per-channel over active pixels only.
                Ll1 = _masked_mean(torch.abs(gt_img_unmasked - pred_img_unmasked), mask)
            else:
                Ll1 = torch.abs(gt_img - pred_img).mean()
            simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        if self.config.loss == "depth":
            loss_dict = {
                "main_loss": Ll1,
                "scale_reg": scale_reg,
            }
        else:
            loss_dict = {
                "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
                "scale_reg": scale_reg,
            }

        # Losses for mcmc
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                mcmc_opacity_reg = (
                    self.config.mcmc_opacity_reg * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
                loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg
            if self.config.mcmc_scale_reg > 0.0:
                mcmc_scale_reg = self.config.mcmc_scale_reg * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

            # Optional: Bezier shell topology losses (Xing + min thickness).
            if bool(getattr(self.config, "bezier_topo_loss_enabled", False)) and hasattr(self, "bezier_shell_cp_out"):
                cp_out = getattr(self, "bezier_shell_cp_out")
                cp_in = getattr(self, "bezier_shell_cp_in")
                try:
                    if bool(getattr(self.config, "bezier_surface_pruning_enabled", False)) and hasattr(self, "bezier_surface_active"):
                        keep_surf = getattr(self, "bezier_surface_active")
                        if bool(keep_surf.any()):
                            cp_out = cp_out[keep_surf]
                            cp_in = cp_in[keep_surf]
                        else:
                            # No active surfaces -> skip topo loss.
                            raise RuntimeError("No active Bezier surfaces for topo loss.")
                    X = sample_paired_bezier_surfaces(
                        cp_out,
                        cp_in,
                        num_r=int(getattr(self.config, "bezier_num_r", 5)),
                        num_u=int(getattr(self.config, "bezier_num_u", 20)),
                        num_v=int(getattr(self.config, "bezier_num_v", 20)),
                    )  # (S,R,U,V,3)
                    topo = bezier_shell_topo_losses_from_samples(
                        X,
                        eps=float(getattr(self.config, "bezier_topo_eps", 1e-6)),
                        delta=float(getattr(self.config, "bezier_topo_delta", 0.0)),
                    )
                    lam_x = float(getattr(self.config, "bezier_topo_lambda_xing", 0.0))
                    lam_t = float(getattr(self.config, "bezier_topo_lambda_thick", 0.0))
                    if lam_x != 0.0:
                        loss_dict["bezier_xing_loss"] = topo["xing"] * lam_x
                    if lam_t != 0.0:
                        loss_dict["bezier_thick_loss"] = topo["thick"] * lam_t
                    if lam_x != 0.0 or lam_t != 0.0:
                        loss_dict["bezier_topo_loss"] = (topo["xing"] * lam_x) + (topo["thick"] * lam_t)
                except Exception as e:
                    # Keep training stable if something goes wrong.
                    CONSOLE.log(f"[yellow]Warning: Bezier topo loss failed (skipping): {e}[/yellow]")

            # Optional: L2 regularizer on open Bezier control points (reduces excessive deformations).
            if (
                hasattr(self, "bezier_open_cp")
                and float(getattr(self.config, "bezier_open_cp_l2_lambda", 0.0)) != 0.0
            ):
                lambda_reg = float(self.config.bezier_open_cp_l2_lambda)
                loss_reg = lambda_reg * (self.bezier_open_cp**2).mean()
                loss_dict["bezier_cp_l2_loss"] = loss_reg

            # Optional: attach (a subset of) Gaussian means to the Bezier shell points they were initialized from.
            if (
                bool(getattr(self.config, "bezier_attach_loss_enabled", False))
                and hasattr(self, "bezier_shell_cp_out")
                and hasattr(self, "bezier_attach_shell_idx")
                and float(getattr(self.config, "bezier_attach_lambda", 0.0)) != 0.0
            ):
                try:
                    stop_step = int(getattr(self.config, "bezier_attach_stop_step", 0))
                    # Guard: only apply early, and only if the Gaussian set hasn't changed size.
                    if stop_step > 0 and int(self.step) <= stop_step:
                        n_init = int(getattr(self, "_bezier_attach_num_points_init", 0))
                        if n_init > 0 and int(self.num_points) == n_init:
                            Xs = sample_paired_bezier_surfaces(
                                getattr(self, "bezier_shell_cp_out"),
                                getattr(self, "bezier_shell_cp_in"),
                                num_r=int(getattr(self.config, "bezier_num_r", 5)),
                                num_u=int(getattr(self.config, "bezier_num_u", 20)),
                                num_v=int(getattr(self.config, "bezier_num_v", 20)),
                            )  # (S,R,U,V,3)
                            ss = getattr(self, "bezier_attach_shell_idx")
                            rr = getattr(self, "bezier_attach_r_idx")
                            uu = getattr(self, "bezier_attach_u_idx")
                            vv = getattr(self, "bezier_attach_v_idx")
                            target = Xs[ss, rr, uu, vv]  # (N,3)
                            means = self.gauss_params["means"]  # (N,3) at this stage
                            attach = ((means - target) ** 2).sum(dim=-1).mean()
                            loss_dict["bezier_attach_loss"] = float(getattr(self.config, "bezier_attach_lambda", 0.0)) * attach
                except Exception as e:
                    CONSOLE.log(f"[yellow]Warning: Bezier attach loss failed (skipping): {e}[/yellow]")

        return loss_dict

    @torch.no_grad()
    def _get_or_compute_da3_depth_fullres(self, cam_idx: int, image: torch.Tensor, focal_px: float) -> torch.Tensor:
        """Compute (once per training image) Depth Anything 3 metric depth from the GT RGB.

        Returns full-resolution [H,W,1] on CPU for caching. During training we then downscale
        this depth to match the current resolution schedule.
        """
        # Controlla se esiste in cache
        if cam_idx >= 0 and cam_idx in self._da3_depth_cache_fullres:
            return self._da3_depth_cache_fullres[cam_idx]

        # Prova a caricare da disco se esiste
        if cam_idx >= 0:
            import cv2
            import numpy as np
            from pathlib import Path

            depth_dir = Path("data/depth")
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth_path = depth_dir / f"depth_{cam_idx:05d}.png"

            if depth_path.exists():
                try:
                    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
                    if depth_mm is not None:
                        # Converti da millimetri a metri
                        depth = torch.from_numpy(depth_mm.astype(np.float32) / 1000.0)[:, :, None]
                        # Cache in memoria
                        self._da3_depth_cache_fullres[cam_idx] = depth
                        return depth
                except Exception as e:
                    CONSOLE.log(f"[yellow]Warning: Could not load DA3 depth from {depth_path}: {e}[/yellow]")

        if focal_px <= 0:
            raise RuntimeError("config.loss='depth' requires a valid focal length in pixels (focal_px_full).")

        rgb = image
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        rgb = rgb.to(self.device)

        # If RGBA, composite deterministically to avoid varying depth targets across steps.
        if rgb.shape[-1] == 4:
            black_bg = torch.zeros(3, device=self.device, dtype=rgb.dtype)
            rgb = self.composite_with_background(rgb, black_bg)
        else:
            rgb = rgb[..., :3]

        try:
            from nerfstudio.models.da3_depth import get_da3_metric_estimator
        except Exception as e:
            raise RuntimeError(
                "config.loss='depth' requires Depth Anything 3. "
                "See nerfstudio/models/da3_depth.py for installation instructions."
            ) from e

        est = get_da3_metric_estimator(
            model_id=self.config.da3_model_id.lower(),
            max_side=self.config.da3_max_side,
            use_half=self.config.da3_use_half,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        depth = est.infer_metric_depth(rgb, focal_px=focal_px)  # [H,W,1]
        depth_cpu = depth.detach().cpu()

        # Salva su disco se cam_idx è valido
        if cam_idx >= 0:
            try:
                import cv2
                import numpy as np
                from pathlib import Path

                depth_dir = Path("data/depth")
                depth_dir.mkdir(parents=True, exist_ok=True)
                depth_path = depth_dir / f"depth_{cam_idx:05d}.png"

                # Converti da metri a millimetri (uint16)
                depth_mm = (depth_cpu.squeeze(-1).numpy() * 1000.0).astype(np.uint16)
                cv2.imwrite(str(depth_path), depth_mm)
                CONSOLE.log(f"[green]Saved DA3 depth to {depth_path}[/green]")
            except Exception as e:
                CONSOLE.log(f"[yellow]Warning: Could not save DA3 depth: {e}[/yellow]")

        # Cache in memoria
        if cam_idx >= 0:
            self._da3_depth_cache_fullres[cam_idx] = depth_cpu
        return depth_cpu

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
