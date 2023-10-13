# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import argparse


#################################################################################
#                            practical argparse utils                           #
#################################################################################

def accelerate_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # Device
    parser.add_argument("-cpu", "--use_cpu", action="store_true",
                        help="Whether or not disable cuda")

    # Gradient Accumulation
    parser.add_argument("-cumgard", "--gradient-accumulate-step",
                        type=int, default=1)
    parser.add_argument("--split-batches", action="store_true",
                        help="Whether or not the accelerator should split the batches "
                             "yielded by the dataloaders across the devices.")

    # Nvidia-Apex and GradScaler
    parser.add_argument("-mprec", "--mixed-precision",
                        type=str, default='no', choices=['no', 'fp16', 'bf16'],
                        help="Whether to use mixed precision. Choose"
                             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
                             "and an Nvidia Ampere GPU.")
    parser.add_argument("--init-scale",
                        type=float, default=65536.0,
                        help="Default value: `2.**16 = 65536.0` ,"
                             "For ImageNet experiments, '2.**20 = 1048576.0' was a good default value."
                             "the others: `2.**17 = 131072.0` ")
    parser.add_argument("--growth-factor", type=float, default=2.0)
    parser.add_argument("--backoff-factor", type=float, default=0.5)
    parser.add_argument("--growth-interval", type=int, default=2000)

    # Gradient Normalization
    parser.add_argument("-gard_norm", "--max_grad_norm", type=float, default=-1)

    # Trackers
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--project-name", type=str, default="SketchGeneration")
    parser.add_argument("--entity", type=str, default="ximinng")
    parser.add_argument("--tensorboard", action="store_true")

    # reproducibility
    parser.add_argument("-d", "--seed", default=42, type=int)

    # result path
    parser.add_argument("-respath", "--results_path",
                        type=str, default="",
                        help="If it is None, it is automatically generated.")

    # timing
    parser.add_argument("-log_step", "--log_step", default=1000, type=int,
                        help="can be use to control log.")
    parser.add_argument("-eval_step", "--eval_step", default=5000, type=int,
                        help="can be use to calculate some metrics.")
    parser.add_argument("-save_step", "--save_step", default=5000, type=int,
                        help="can be use to control saving checkpoint.")

    # update configuration interface
    # example: python main.py -c main.yaml -update "nnet.depth=16 batch_size=16"
    parser.add_argument("-update",
                        type=str, default=None,
                        help="modified hyper-parameters of config file.")
    return parser


def ema_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ema', action='store_true', help='enable EMA model')
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=100)
    parser.add_argument("--ema_update_every", type=int, default=10)
    return parser


def base_data_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-spl", "--split",
                        default='test', type=str,
                        choices=['train', 'val', 'test', 'all'],
                        help="which part of the data set, 'all' means combine training and test sets.")
    parser.add_argument("-j", "--num_workers",
                        default=6, type=int,
                        help="how many subprocesses to use for data loading.")
    parser.add_argument("--shuffle",
                        action='store_true',
                        help="how many subprocesses to use for data loading.")
    parser.add_argument("--drop_last",
                        action='store_true',
                        help="how many subprocesses to use for data loading.")
    return parser


def base_training_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-tbz", "--train_batch_size",
                        default=32, type=int,
                        help="how many images to sample during training.")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-wd", "--weight_decay", default=0, type=float)
    return parser


def base_sampling_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-vbz", "--valid_batch_size",
                        default=1, type=int,
                        help="how many images to sample during evaluation")
    parser.add_argument("-ts", "--total_samples",
                        default=2000, type=int,
                        help="the total number of samples, can be used to calculate FID.")
    parser.add_argument("-ns", "--num_samples",
                        default=4, type=int,
                        help="number of samples taken at a time, "
                             "can be used to repeatedly induce samples from a generation model "
                             "from a fixed guided information, "
                             "eg: `one latent to ns samples` (1 latent to 5 photo generation) ")
    return parser
