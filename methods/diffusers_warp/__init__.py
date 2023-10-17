# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
from typing import AnyStr
import pathlib
from collections import OrderedDict
from packaging import version

import torch
from diffusers import StableDiffusionPipeline, SchedulerMixin, DiffusionPipeline
from diffusers.utils import is_torch_version, is_xformers_available

huggingface_model_dict = OrderedDict({
    "sd14": "CompVis/stable-diffusion-v1-4",  # resolution: 512
    "sd15": "runwayml/stable-diffusion-v1-5",  # resolution: 512
    "sd21b": "stabilityai/stable-diffusion-2-1-base",  # resolution: 512
    "sd21": "stabilityai/stable-diffusion-2-1",  # resolution: 768
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",  # resolution: 1024
})

_model2resolution = {
    "sd14": 512,
    "sd15": 512,
    "sd21b": 512,
    "sd21": 768,
    "sdxl": 1024,
}


def model2res(model_id: str):
    return _model2resolution.get(model_id, 512)


def init_diffusion_pipeline(model_id: AnyStr,
                            custom_pipeline: StableDiffusionPipeline,
                            custom_scheduler: SchedulerMixin = None,
                            device: torch.device = "cuda",
                            torch_dtype: torch.dtype = torch.float32,
                            local_files_only: bool = True,
                            force_download: bool = False,
                            resume_download: bool = False,
                            ldm_speed_up: bool = False,
                            enable_xformers: bool = True,
                            gradient_checkpoint: bool = False,
                            lora_path: AnyStr = None,
                            unet_path: AnyStr = None) -> StableDiffusionPipeline:
    """
    A tool for initial diffusers model.

    Args:
        model_id (`str` or `os.PathLike`, *optional*): pretrained_model_name_or_path
        custom_pipeline: any StableDiffusionPipeline pipeline
        custom_scheduler: any scheduler
        device: set device
        local_files_only: prohibited download model
        force_download: forced download model
        resume_download: re-download model
        ldm_speed_up: use the `torch.compile` api to speed up unet
        enable_xformers: enable memory efficient attention from [xFormers]
        gradient_checkpoint: activates gradient checkpointing for the current model
        lora_path: load LoRA checkpoint
        unet_path: load unet checkpoint

    Returns:
            diffusers.StableDiffusionPipeline
    """

    # get model id
    model_id = huggingface_model_dict.get(model_id, model_id)

    # process diffusion model
    if custom_scheduler is not None:
        pipeline = custom_pipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            force_download=force_download,
            resume_download=resume_download,
            scheduler=custom_scheduler.from_pretrained(model_id,
                                                       subfolder="scheduler",
                                                       local_files_only=local_files_only)
        ).to(device)
    else:
        pipeline = custom_pipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            force_download=force_download,
            resume_download=resume_download,
        ).to(device)

    # process unet model if exist
    if unet_path is not None and pathlib.Path(unet_path).exists():
        print(f"=> load u-net from {unet_path}")
        pipeline.unet.from_pretrained(model_id, subfolder="unet")

    # process lora layers if exist
    if lora_path is not None and pathlib.Path(lora_path).exists():
        pipeline.unet.load_attn_procs(lora_path)
        print(f"=> load lora layers into U-Net from {lora_path} ...")

    # torch.compile
    if ldm_speed_up:
        if is_torch_version(">=", "2.0.0"):
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
            print(f"=> enable torch.compile on U-Net")
        else:
            print(f"=> warning: calling torch.compile speed-up failed, since torch version <= 2.0.0")

    # Meta xformers
    if enable_xformers:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems during training, please update xFormers to at least 0.0.17. "
                    "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            print(f"=> enable xformers")
            pipeline.unet.enable_xformers_memory_efficient_attention()
        else:
            print(f"=> warning: calling xformers failed")

    # gradient checkpointing
    if gradient_checkpoint:
        if pipeline.unet.is_gradient_checkpointing:
            print(f"=> enable gradient checkpointing")
            pipeline.unet.enable_gradient_checkpointing()
        else:
            print("=> waring: gradient checkpointing is not activated for this model.")

    print(f"Diffusion Model: {model_id}")
    print(pipeline.scheduler)
    return pipeline
