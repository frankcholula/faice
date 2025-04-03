# -*- coding: UTF-8 -*-
"""
@Time : 03/04/2025 06:17
@Author : xiaoguangliang
@File : tools.py
@Project : faice
"""
import time
from contextlib import contextmanager


@contextmanager
def timer(msg='all tasks'):
    """
    Calculate the time of running
    @return:
    """
    startTime = time.time()
    yield
    endTime = time.time()
    # print(f'The time cost for {msg}：{round(1000.0 * (endTime - startTime), 2)}, ms')
    print(f'The time cost for {msg}：', round(endTime - startTime, 2), 's')
