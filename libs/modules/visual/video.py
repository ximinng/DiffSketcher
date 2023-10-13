# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
from typing import Any, Union
import pathlib

import cv2


def create_video(num_iter: int,
                 save_dir: Union[Any, pathlib.Path],
                 video_frame_freq: int = 1,
                 fname: str = "rendering_process",
                 verbose: bool = True):
    if not isinstance(save_dir, pathlib.Path):
        save_dir = pathlib.Path(save_dir)

    img_array = []
    for i in range(0, num_iter):
        if i % video_frame_freq == 0 or i == num_iter - 1:
            filename = save_dir / f"iter{i}.png"
            img = cv2.imread(filename.as_posix())
            img_array.append(img)

    video_name = save_dir / f"{fname}.mp4"
    out = cv2.VideoWriter(
        video_name.as_posix(),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30.0,  # fps
        (600, 600)  # video size
    )
    for iii in range(len(img_array)):
        out.write(img_array[iii])
    out.release()

    if verbose:
        print(f"video saved in '{video_name}'.")
