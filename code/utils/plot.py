# -*- coding: UTF-8 -*-
"""
@Time : 23/04/2025 16:40
@Author : xiaoguangliang
@File : plot.py
@Project : code
"""
from matplotlib import pyplot
from typing import List, Optional
import numpy
import math
import torch
import os


def plot_images(
        images: 'torch.Tensor',
        titles: Optional[List[str]] = None,
        fig_titles: Optional[str] = None,
        save_title: Optional[str] = None,
        save_dir: Optional[str] = None,
        cols: int = 4):
    _images = images
    b, c, h, w = _images.shape

    # Copy a channel to 3 channels, when there is only one channel
    if c == 1:
        _images = torch.repeat_interleave(_images, 3, dim=1)
    if c > 3:
        _images = _images[:, :3, :, :]

    # Calculate the number of rows and columns
    COLS = cols
    ROWS = int(math.ceil(b / COLS))

    if torch.is_tensor(images):
        images = _images.detach().cpu().numpy()

    _images = numpy.transpose(images, [0, 2, 3, 1])
    fig, axes = pyplot.subplots(ROWS, COLS, figsize=(COLS, ROWS))
    pyplot.subplots_adjust(wspace=0.05, hspace=0.05)

    if fig_titles is not None:
        fig.suptitle(fig_titles, fontsize=10)

    if titles is None:
        titles = ["" for _ in range(b)]

    axes = axes.flatten()

    assert len(titles) == b
    assert b <= axes.size

    for image, axis, title in zip(_images, axes, titles):
        axis.imshow(image)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.set_title(title, fontsize=8)

    # Save plot as image
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        from datetime import datetime
        _title = save_title if save_title is not None else "no title"
        save_path = os.path.join(save_dir,
                                 _title + "_" + datetime.now().strftime(r"%m_%d %H_%M_%S") + ".png")
        pyplot.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        pyplot.show()


def plot_arrays(x: 'torch.Tensor', ys: 'torch.Tensor'):
    _, axes = pyplot.subplots(1, 1)
    for y in ys:
        axes.plot(x.cpu().numpy(), y.cpu().numpy(), markersize=3)

    pyplot.show()
