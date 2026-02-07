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
Abstracts for the Pipeline class.
"""

from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
import torch.distributed as dist
import torchvision.utils as vutils
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the data manager and model classes from the trainer,
    worrying about:
    1) Fetching data with the data manager
    2) Feeding the model the data and fetching the loss
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'train': loads train/eval datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    datamanager: DataManager
    _model: Model
    world_size: int

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.
        """

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """

    @abstractmethod
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """


@dataclass
class VanillaPipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VanillaPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""


class VanillaPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata  # type: ignore
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]  # type: ignore
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]  # type: ignore
            seed_pts = (pts, pts_rgb)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # Optional: SAM2-based semantic labels for COLMAP seed points (clean init-time labeling).
        model_metadata: Dict[str, Any] = dict(self.datamanager.train_dataset.metadata)
        sam2_enabled = bool(getattr(config.model, "sam2_semantic_init_enabled", False))
        if sam2_enabled and seed_pts is not None and hasattr(self.datamanager, "train_dataparser_outputs"):
            try:
                from nerfstudio.models.sam2_semantics import (
                    GroundedSAM2SemanticInitConfig,
                    SAM2SemanticInitConfig,
                    compute_seed_semantic_labels_from_grounded_sam2_all_images,
                    compute_seed_semantic_labels_and_labelmap_from_sam2,
                )

                # Prefer text-prompt mode if prompts are provided.
                text_prompts = getattr(config.model, "sam2_text_prompts", None)
                if text_prompts is not None and len(text_prompts) > 0:
                    image_indices = getattr(config.model, "sam2_segment_image_indices", None)
                    if image_indices is None and (not bool(getattr(config.model, "sam2_segment_all_train_images", True))):
                        image_indices = [int(getattr(config.model, "sam2_init_image_idx", 0))]

                    grounded_cfg = GroundedSAM2SemanticInitConfig(
                        model_id=str(getattr(config.model, "sam2_model_id", "facebook/sam2-hiera-large")),
                        label_id=int(getattr(config.model, "sam2_label_id", 1)),
                        text_prompts=list(text_prompts),
                        groundingdino_model_id=str(
                            getattr(config.model, "sam2_groundingdino_model_id", "IDEA-Research/grounding-dino-base")
                        ),
                        groundingdino_revision=getattr(config.model, "sam2_groundingdino_revision", None),
                        groundingdino_config_filename=getattr(config.model, "sam2_groundingdino_config_filename", None),
                        groundingdino_checkpoint_filename=getattr(
                            config.model, "sam2_groundingdino_checkpoint_filename", None
                        ),
                        groundingdino_config_path=getattr(config.model, "sam2_groundingdino_config_path", None),
                        groundingdino_checkpoint_path=getattr(config.model, "sam2_groundingdino_checkpoint_path", None),
                        box_threshold=float(getattr(config.model, "sam2_groundingdino_box_threshold", 0.30)),
                        text_threshold=float(getattr(config.model, "sam2_groundingdino_text_threshold", 0.25)),
                        max_boxes_per_prompt=int(getattr(config.model, "sam2_groundingdino_max_boxes_per_prompt", 8)),
                        image_indices=image_indices,
                        output_dir=str(getattr(config.model, "sam2_segmentation_output_dir", "data/grounded_sam2")),
                        save_labelmap_npy=bool(getattr(config.model, "sam2_save_labelmap_npy", True)),
                        save_per_prompt_masks=bool(getattr(config.model, "sam2_save_per_prompt_masks", False)),
                        mask_distance_px=int(getattr(config.model, "sam2_mask_distance_px", 0)),
                        device=str(getattr(config.model, "sam2_device", "cuda"))
                        if getattr(config.model, "sam2_device", None)
                        else None,
                        groundingdino_device=str(getattr(config.model, "sam2_device", "cuda"))
                        if getattr(config.model, "sam2_device", None)
                        else None,
                    )

                    labels, label_names = compute_seed_semantic_labels_from_grounded_sam2_all_images(
                        train_dataset=self.datamanager.train_dataset,
                        train_dataparser_outputs=self.datamanager.train_dataparser_outputs,  # type: ignore
                        config=grounded_cfg,
                    )
                    model_metadata["seed_semantic_labels"] = labels
                    model_metadata["seed_semantic_label_names"] = label_names
                    nlab = int(torch.unique(labels[labels > 0]).numel())
                    CONSOLE.log(
                        f"[green]GroundedSAM2 semantic init: labeled {int((labels > 0).sum().item())} / {int(labels.numel())} seed points (labels={nlab})[/green]"
                    )
                    CONSOLE.log(
                        f"[green]Saved per-image segmentations to {getattr(config.model, 'sam2_segmentation_output_dir', 'data/grounded_sam2')}[/green]"
                    )
                    
                    # Convert SAM2 labelmaps to binary masks for dataparser
                    try:
                        from nerfstudio.models.sam2_semantics import convert_sam2_labelmaps_to_binary_masks
                        from pathlib import Path
                        
                        sam2_output_dir = str(getattr(config.model, "sam2_segmentation_output_dir", "data/grounded_sam2"))
                        # Get dataparser config from datamanager (after setup)
                        dataparser_config = getattr(self.datamanager, "dataparser_config", None)
                        if dataparser_config is None:
                            dataparser_config = getattr(config.datamanager, "dataparser", None)
                        
                        masks_output_dir = "masks"  # default
                        data_path = Path(".")
                        images_path = "images"
                        downscale_factor = None
                        downscale_rounding_mode = "floor"
                        if dataparser_config is not None:
                            masks_output_dir = str(getattr(dataparser_config, "masks_path", None) or "masks")
                            data_path = Path(getattr(dataparser_config, "data", None) or Path("."))
                            images_path = str(getattr(dataparser_config, "images_path", None) or "images")
                            downscale_factor = getattr(dataparser_config, "downscale_factor", None)
                            downscale_rounding_mode = str(getattr(dataparser_config, "downscale_rounding_mode", "floor") or "floor")

                        # Prefer the actual runtime downscale factor from the dataparser instance (handles auto factor).
                        try:
                            dp = getattr(self.datamanager, "dataparser", None)
                            actual_df = getattr(dp, "_downscale_factor", None)
                            if actual_df is not None:
                                downscale_factor = int(actual_df)
                        except Exception:
                            pass
                        
                        num_masks = convert_sam2_labelmaps_to_binary_masks(
                            sam2_output_dir=sam2_output_dir,
                            masks_output_dir=masks_output_dir,
                            train_dataparser_outputs=self.datamanager.train_dataparser_outputs,  # type: ignore
                            data_path=data_path,
                            images_path=images_path,
                            downscale_factor=downscale_factor,
                            downscale_rounding_mode=downscale_rounding_mode,
                        )
                        if num_masks > 0:
                            CONSOLE.log(
                                f"[green]Converted {num_masks} SAM2 labelmaps to binary masks in {data_path / masks_output_dir}[/green]"
                            )
                            CONSOLE.log(
                                f"[green]To use these masks, set --masks-path {masks_output_dir} in your train command[/green]"
                            )
                        else:
                            CONSOLE.log(
                                f"[yellow]No SAM2 labelmaps found in {sam2_output_dir} to convert[/yellow]"
                            )
                    except Exception as e:
                        CONSOLE.log(f"[yellow]Warning: Could not convert SAM2 labelmaps to masks: {e}[/yellow]")
                else:
                    sam2_cfg = SAM2SemanticInitConfig(
                        model_id=str(getattr(config.model, "sam2_model_id", "facebook/sam2-hiera-large")),
                        image_idx=int(getattr(config.model, "sam2_init_image_idx", 0)),
                        label_id=int(getattr(config.model, "sam2_label_id", 1)),
                        point_coords=getattr(config.model, "sam2_point_coords", None),
                        point_labels=getattr(config.model, "sam2_point_labels", None),
                        box_xyxy=getattr(config.model, "sam2_box_xyxy", None),
                        mask_distance_px=int(getattr(config.model, "sam2_mask_distance_px", 0)),
                        auto_grid_stride=int(getattr(config.model, "sam2_auto_grid_stride", 32)),
                        auto_max_masks=int(getattr(config.model, "sam2_auto_max_masks", 64)),
                        auto_min_mask_area=int(getattr(config.model, "sam2_auto_min_mask_area", 256)),
                        auto_dedup_iou_thresh=float(getattr(config.model, "sam2_auto_dedup_iou_thresh", 0.9)),
                        device=str(getattr(config.model, "sam2_device", "cuda"))
                        if getattr(config.model, "sam2_device", None)
                        else None,
                    )
                    labels, label_map, image_uint8 = compute_seed_semantic_labels_and_labelmap_from_sam2(
                        train_dataset=self.datamanager.train_dataset,
                        train_dataparser_outputs=self.datamanager.train_dataparser_outputs,  # type: ignore
                        config=sam2_cfg,
                    )
                    model_metadata["seed_semantic_labels"] = labels
                    nobj = int(torch.unique(labels[labels > 0]).numel())
                    CONSOLE.log(
                        f"[green]SAM2 semantic init: labeled {int((labels > 0).sum().item())} / {int(labels.numel())} seed points (objects={nobj})[/green]"
                    )

                    # Save visualization (label map + overlay) so it is easy to inspect.
                    try:
                        import numpy as np
                        from pathlib import Path
                        from PIL import Image

                        out_dir = Path("data/sam2")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        idx = int(sam2_cfg.image_idx)

                        # Colorize label map.
                        lm = label_map.detach().cpu().numpy().astype(np.int32)
                        H, W = lm.shape
                        color = np.zeros((H, W, 3), dtype=np.uint8)
                        unique = np.unique(lm)
                        # Deterministic palette per label.
                        rng = np.random.default_rng(0)
                        palette = {}
                        for lab in unique:
                            if lab <= 0:
                                continue
                            palette[int(lab)] = rng.integers(low=0, high=255, size=(3,), dtype=np.uint8)
                        for lab, col in palette.items():
                            color[lm == lab] = col

                        label_png = out_dir / f"labelmap_{idx:05d}.png"
                        Image.fromarray(color).save(label_png)

                        img = image_uint8
                        if img.shape[2] == 4:
                            img = img[:, :, :3]
                        img = img.astype(np.uint8, copy=False)
                        overlay = img.copy()
                        mask = lm > 0
                        alpha = 0.55
                        overlay[mask] = (overlay[mask] * (1.0 - alpha) + color[mask] * alpha).astype(np.uint8)
                        overlay_png = out_dir / f"overlay_{idx:05d}.png"
                        Image.fromarray(overlay).save(overlay_png)

                        CONSOLE.log(f"[green]Saved SAM2 label map to {label_png}[/green]")
                        CONSOLE.log(f"[green]Saved SAM2 overlay to {overlay_png}[/green]")
                    except Exception as e:
                        CONSOLE.log(f"[yellow]Warning: could not save SAM2 visualization: {e}[/yellow]")
                    
                    # Convert SAM2 labelmap to binary mask for dataparser (single image mode)
                    try:
                        from nerfstudio.models.sam2_semantics import convert_sam2_labelmaps_to_binary_masks
                        from pathlib import Path
                        
                        sam2_output_dir = str(getattr(config.model, "sam2_segmentation_output_dir", "data/grounded_sam2"))
                        # Get dataparser config from datamanager (after setup)
                        dataparser_config = getattr(self.datamanager, "dataparser_config", None)
                        if dataparser_config is None:
                            dataparser_config = getattr(config.datamanager, "dataparser", None)
                        
                        masks_output_dir = "masks"  # default
                        data_path = Path(".")
                        images_path = "images"
                        downscale_factor = None
                        downscale_rounding_mode = "floor"
                        if dataparser_config is not None:
                            masks_output_dir = str(getattr(dataparser_config, "masks_path", None) or "masks")
                            data_path = Path(getattr(dataparser_config, "data", None) or Path("."))
                            images_path = str(getattr(dataparser_config, "images_path", None) or "images")
                            downscale_factor = getattr(dataparser_config, "downscale_factor", None)
                            downscale_rounding_mode = str(getattr(dataparser_config, "downscale_rounding_mode", "floor") or "floor")

                        # Prefer the actual runtime downscale factor from the dataparser instance (handles auto factor).
                        try:
                            dp = getattr(self.datamanager, "dataparser", None)
                            actual_df = getattr(dp, "_downscale_factor", None)
                            if actual_df is not None:
                                downscale_factor = int(actual_df)
                        except Exception:
                            pass
                        
                        num_masks = convert_sam2_labelmaps_to_binary_masks(
                            sam2_output_dir=sam2_output_dir,
                            masks_output_dir=masks_output_dir,
                            train_dataparser_outputs=self.datamanager.train_dataparser_outputs,  # type: ignore
                            data_path=data_path,
                            images_path=images_path,
                            downscale_factor=downscale_factor,
                            downscale_rounding_mode=downscale_rounding_mode,
                        )
                        if num_masks > 0:
                            CONSOLE.log(
                                f"[green]Converted {num_masks} SAM2 labelmap(s) to binary masks in {data_path / masks_output_dir}[/green]"
                            )
                            CONSOLE.log(
                                f"[green]To use these masks, set --masks-path {masks_output_dir} in your train command[/green]"
                            )
                    except Exception as e:
                        CONSOLE.log(f"[yellow]Warning: Could not convert SAM2 labelmap to mask: {e}[/yellow]")
            except Exception as e:
                CONSOLE.log(f"[yellow]SAM2 semantic init skipped: {e}[/yellow]")

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=model_metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            for camera, batch in data_loader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(
                            image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.png"
                        )

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        self.train()
        return metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Get the average metrics for evaluation images."""
        assert hasattr(self.datamanager, "fixed_indices_eval_dataloader"), (
            "datamanager must have 'fixed_indices_eval_dataloader' attribute"
        )
        image_prefix = "eval"
        return self.get_average_image_metrics(
            self.datamanager.fixed_indices_eval_dataloader,  # type: ignore
            image_prefix,
            step,
            output_path,
            get_std,
        )

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
