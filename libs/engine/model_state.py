# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import gc
from functools import partial
from typing import Union, List
from pathlib import Path
from datetime import datetime, timedelta

from omegaconf import DictConfig
from pprint import pprint
import torch
from accelerate.utils import LoggerType
from accelerate import (
    Accelerator,
    GradScalerKwargs,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs
)

from ..modules.ema import EMA
from ..utils.logging import get_logger


class ModelState:
    """
    Handling logger and `hugging face` accelerate training

    features:
        - Mixed Precision
        - Gradient Scaler
        - Gradient Accumulation
        - Optimizer
        - EMA
        - Logger (default: python print)
        - Monitor (default: wandb, tensorboard)
    """

    def __init__(
            self,
            args,
            log_path_suffix: str = None,
            ignore_log=False,  # whether to create log file or not
    ) -> None:
        self.args: DictConfig = args

        """check valid"""
        mixed_precision = self.args.get("mixed_precision")
        # Bug: omegaconf convert 'no' to false
        mixed_precision = "no" if type(mixed_precision) == bool else mixed_precision
        split_batches = self.args.get("split_batches", False)
        gradient_accumulate_step = self.args.get("gradient_accumulate_step", 1)
        assert gradient_accumulate_step >= 1, f"except gradient_accumulate_step >= 1, get {gradient_accumulate_step}"

        """create working space"""
        # rule: ['./config'. 'method_name', 'exp_name.yaml']
        # -> results_path: ./runs/{method_name}-{exp_name}, as a base folder
        # config_prefix, config_name = str(self.args.get("config")).split('/')
        # config_name_only = str(config_name).split(".")[0]

        config_name_only = str(self.args.get("config")).split(".")[0]
        results_folder = self.args.get("results_path", None)
        if results_folder is None:
            # self.results_path = Path("./workdir") / f"{config_prefix}-{config_name_only}"
            self.results_path = Path("./workdir") / f"{config_name_only}"
        else:
            # self.results_path = Path(results_folder) / f"{config_prefix}-{config_name_only}"
            self.results_path = Path(results_folder) / f"{config_name_only}"

        # update results_path: ./runs/{method_name}-{exp_name}/{log_path_suffix}
        # noting: can be understood as "results dir / methods / ablation study / your result"
        if log_path_suffix is not None:
            self.results_path = self.results_path / log_path_suffix

        kwargs_handlers = []
        """mixed precision training"""
        if args.mixed_precision == "no":
            scaler_handler = GradScalerKwargs(
                init_scale=args.init_scale,
                growth_factor=args.growth_factor,
                backoff_factor=args.backoff_factor,
                growth_interval=args.growth_interval,
                enabled=True
            )
            kwargs_handlers.append(scaler_handler)

        """distributed training"""
        ddp_handler = DistributedDataParallelKwargs(
            dim=0,
            broadcast_buffers=True,
            static_graph=False,
            bucket_cap_mb=25,
            find_unused_parameters=False,
            check_reduction=False,
            gradient_as_bucket_view=False
        )
        kwargs_handlers.append(ddp_handler)

        init_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=1200))
        kwargs_handlers.append(init_handler)

        """init visualized tracker"""
        log_with = []
        self.args.visual = False
        if args.use_wandb:
            log_with.append(LoggerType.WANDB)
        if args.tensorboard:
            log_with.append(LoggerType.TENSORBOARD)

        """hugging face Accelerator"""
        self.accelerator = Accelerator(
            device_placement=True,
            split_batches=split_batches,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulate_step,
            cpu=True if args.use_cpu else False,
            log_with=None if len(log_with) == 0 else log_with,
            project_dir=self.results_path / "vis",
            kwargs_handlers=kwargs_handlers,
        )

        """logs"""
        if self.accelerator.is_local_main_process:
            # for logging results in a folder periodically
            self.results_path.mkdir(parents=True, exist_ok=True)
            if not ignore_log:
                now_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
                self.logger = get_logger(
                    logs_dir=self.results_path.as_posix(),
                    file_name=f"{now_time}-log-{args.seed}.txt"
                )

            print("==> command line args: ")
            print(args.cmd_args)
            print("==> yaml config args: ")
            print(args.yaml_config)

            print("\n***** Model State *****")
            if self.accelerator.distributed_type != "NO":
                print(f"-> Distributed Type: {self.accelerator.distributed_type}")
            # print(f"-> Split Batch Size: {split_batches}, Total Batch Size: {self.actual_batch_size}")
            print(f"-> Mixed Precision: {mixed_precision}, AMP: {self.accelerator.native_amp},"
                  f" Gradient Accumulate Step: {gradient_accumulate_step}")
            print(f"-> Weight dtype:  {self.weight_dtype}")

            if self.accelerator.scaler_handler is not None and self.accelerator.scaler_handler.enabled:
                print(f"-> Enabled GradScaler: {self.accelerator.scaler_handler.to_kwargs()}")

            if args.use_wandb:
                print(f"-> Init trackers: 'wandb' ")
                self.args.visual = True
                self.__init_tracker(project_name="my_project", tags=None, entity="")

            print(f"-> Working Space: '{self.results_path}'")

        """EMA"""
        self.use_ema = args.get('ema', False)
        self.ema_wrapper = self.__build_ema_wrapper()

        """glob step"""
        self.step = 0

        """log process"""
        self.accelerator.wait_for_everyone()
        print(f'Process {self.accelerator.process_index} using device: {self.accelerator.device}')

        self.print("-> state initialization complete \n")

    def __init_tracker(self, project_name, tags, entity):
        self.accelerator.init_trackers(
            project_name=project_name,
            config=dict(self.args),
            init_kwargs={
                "wandb": {
                    "notes": "accelerate trainer pipeline",
                    "tags": [
                        f"total batch_size: {self.actual_batch_size}"
                    ],
                    "entity": entity,
                }}
        )

    def __build_ema_wrapper(self):
        if self.use_ema:
            self.print(f"-> EMA: {self.use_ema}, decay: {self.args.ema_decay}, "
                       f"update_after_step: {self.args.ema_update_after_step}, "
                       f"update_every: {self.args.ema_update_every}")
            ema_wrapper = partial(
                EMA, beta=self.args.ema_decay,
                update_after_step=self.args.ema_update_after_step,
                update_every=self.args.ema_update_every
            )
        else:
            ema_wrapper = None

        return ema_wrapper

    @property
    def device(self):
        return self.accelerator.device

    @property
    def weight_dtype(self):
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        return weight_dtype

    @property
    def actual_batch_size(self):
        if self.accelerator.split_batches is False:
            actual_batch_size = self.args.batch_size * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps
        else:
            assert self.actual_batch_size % self.accelerator.num_processes == 0
            actual_batch_size = self.args.batch_size
        return actual_batch_size

    @property
    def n_gpus(self):
        return self.accelerator.num_processes

    @property
    def no_decay_params_names(self):
        no_decay = [
            "bn", "LayerNorm", "GroupNorm",
        ]
        return no_decay

    def no_decay_params(self, model, weight_decay):
        """optimization tricks"""
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in self.no_decay_params_names)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in self.no_decay_params_names)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def optimized_params(self, model: torch.nn.Module, verbose=True) -> List:
        """return parameters if `requires_grad` is True

        Args:
            model: pytorch models
            verbose: log optimized parameters

        Examples:
            >>> self.params_optimized = self.optimized_params(uvit, verbose=True)
            >>> optimizer = torch.optim.AdamW(self.params_optimized, lr=args.lr)

        Returns:
                a list of parameters
        """
        params_optimized = []
        for key, value in model.named_parameters():
            if value.requires_grad:
                params_optimized.append(value)
                if verbose:
                    self.print("\t {}, {}, {}".format(key, value.numel(), value.shape))
        return params_optimized

    def save_everything(self, fpath: str):
        """Saving and loading the model, optimizer, RNG generators, and the GradScaler."""
        if not self.accelerator.is_main_process:
            return
        self.accelerator.save_state(fpath)

    def load_save_everything(self, fpath: str):
        """Loading the model, optimizer, RNG generators, and the GradScaler."""
        self.accelerator.load_state(fpath)

    def save(self, milestone: Union[str, float, int], checkpoint: object) -> None:
        if not self.accelerator.is_main_process:
            return

        torch.save(checkpoint, self.results_path / f'model-{milestone}.pt')

    def save_in(self, root: Union[str, Path], checkpoint: object) -> None:
        if not self.accelerator.is_main_process:
            return

        torch.save(checkpoint, root)

    def load_ckpt_model_only(self, model: torch.nn.Module, path: Union[str, Path], rm_module_prefix: bool = False):
        ckpt = torch.load(path, map_location=self.accelerator.device)

        unwrapped_model = self.accelerator.unwrap_model(model)
        if rm_module_prefix:
            unwrapped_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
        else:
            unwrapped_model.load_state_dict(ckpt)
        return unwrapped_model

    def load_shared_weights(self, model: torch.nn.Module, path: Union[str, Path]):
        ckpt = torch.load(path, map_location=self.accelerator.device)
        self.print(f"pretrained_dict len: {len(ckpt)}")
        unwrapped_model = self.accelerator.unwrap_model(model)
        model_dict = unwrapped_model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        unwrapped_model.load_state_dict(model_dict, strict=False)
        self.print(f"selected pretrained_dict: {len(model_dict)}")
        return unwrapped_model

    def print(self, *args, **kwargs):
        """Use in replacement of `print()` to only print once per server."""
        self.accelerator.print(*args, **kwargs)

    def pretty_print(self, msg):
        if self.accelerator.is_local_main_process:
            pprint(dict(msg))

    def close_tracker(self):
        self.accelerator.end_training()

    def free_memory(self):
        self.accelerator.clear()

    def close(self, msg: str = "Training complete."):
        """Use in end of training."""
        self.free_memory()

        if torch.cuda.is_available():
            self.print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
        if self.args.visual:
            self.close_tracker()
        self.print(msg)
