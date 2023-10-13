# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import os
from typing import Tuple
from functools import reduce

from argparse import Namespace
from omegaconf import DictConfig, OmegaConf


#################################################################################
#                             merge yaml and argparse                           #
#################################################################################

def register_resolver():
    OmegaConf.register_new_resolver(
        "add", lambda *numbers: sum(numbers)
    )
    OmegaConf.register_new_resolver(
        "multiply", lambda *numbers: reduce(lambda x, y: x * y, numbers)
    )
    OmegaConf.register_new_resolver(
        "sub", lambda n1, n2: n1 - n2
    )


def _merge_args_and_config(
        cmd_args: Namespace,
        yaml_config: DictConfig,
        read_only: bool = False
) -> Tuple[DictConfig, DictConfig, DictConfig]:
    # convert cmd line args to OmegaConf
    cmd_args_dict = vars(cmd_args)
    cmd_args_list = []
    for k, v in cmd_args_dict.items():
        cmd_args_list.append(f"{k}={v}")
    cmd_args_conf = OmegaConf.from_cli(cmd_args_list)

    # The following overrides the previous configuration
    # cmd_args_list > configs
    args_ = OmegaConf.merge(yaml_config, cmd_args_conf)

    if read_only:
        OmegaConf.set_readonly(args_, True)

    return args_, cmd_args_conf, yaml_config


def merge_configs(args, method_cfg_path):
    """merge command line args (argparse) and config file (OmegaConf)"""
    yaml_config_path = os.path.join("./", "config", method_cfg_path)
    try:
        yaml_config = OmegaConf.load(yaml_config_path)
    except FileNotFoundError as e:
        print(f"error: {e}")
        print(f"input file path: `{method_cfg_path}`")
        print(f"config path: `{yaml_config_path}` not found.")
        raise FileNotFoundError(e)
    return _merge_args_and_config(args, yaml_config, read_only=False)


def update_configs(source_args, update_nodes, strict=True, remove_update_nodes=True):
    """update config file (OmegaConf) with dotlist"""
    if update_nodes is None:
        return source_args

    update_args_list = str(update_nodes).split()
    if len(update_args_list) < 1:
        return source_args

    # check update_args
    for item in update_args_list:
        item_key_ = str(item).split('=')[0]  # get key
        # item_val_ = str(item).split('=')[1]  # get value

        if strict:
            # Tests if a key is existing
            # assert OmegaConf.select(source_args, item_key_) is not None, f"{item_key_} is not existing."

            # Tests if a value is missing
            assert not OmegaConf.is_missing(source_args, item_key_), f"the value of {item_key_} is missing."

            # if keys is None, then add key and set the value
            if OmegaConf.select(source_args, item_key_) is None:
                source_args.item_key_ = item_key_

    # update original yaml params
    update_nodes = OmegaConf.from_dotlist(update_args_list)
    merged_args = OmegaConf.merge(source_args, update_nodes)

    # remove update_args
    if remove_update_nodes:
        OmegaConf.update(merged_args, 'update', '')
    return merged_args


def update_if_exist(source_args, update_nodes):
    """update config file (OmegaConf) with dotlist"""
    if update_nodes is None:
        return source_args

    upd_args_list = str(update_nodes).split()
    if len(upd_args_list) < 1:
        return source_args

    update_args_list = []
    for item in upd_args_list:
        item_key_ = str(item).split('=')[0]  # get key

        # if a key is existing
        # if OmegaConf.select(source_args, item_key_) is not None:
        #     update_args_list.append(item)

        update_args_list.append(item)

    # update source_args if key be selected
    if len(update_args_list) < 1:
        merged_args = source_args
    else:
        update_nodes = OmegaConf.from_dotlist(update_args_list)
        merged_args = OmegaConf.merge(source_args, update_nodes)

    return merged_args


def merge_and_update_config(args):
    register_resolver()

    # if yaml_config is existing, then merge command line args and yaml_config
    # if os.path.isfile(args.config) and args.config is not None:
    if args.config is not None and str(args.config).endswith('.yaml'):
        merged_args, cmd_args, yaml_config = merge_configs(args, args.config)
    else:
        merged_args, cmd_args, yaml_config = args, args, None

    # update the yaml_config with the cmd '-update' flag
    update_nodes = args.update
    final_args = update_configs(merged_args, update_nodes)

    # to simplify log output, we empty this
    yaml_config_update = update_if_exist(yaml_config, update_nodes)
    cmd_args_update = update_if_exist(cmd_args, update_nodes)
    cmd_args_update.update = ""  # clear update params

    final_args.yaml_config = yaml_config_update
    final_args.cmd_args = cmd_args_update

    # update seed
    if final_args.seed < 0:
        import random
        final_args.seed = random.randint(0, 65535)

    return final_args
