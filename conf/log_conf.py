# -*- coding: UTF-8 -*-
"""
@Time : 22/03/2025 09:10
@Author : xiaoguangliang
@File : log_conf.py
@Project : faice
"""
from loguru import logger

from conf.global_setting import SETTINGS

logger.add(SETTINGS.debug_log_file_path, rotation="50 MB", level="DEBUG", format=SETTINGS.log_format,
           enqueue=True)
logger.add(SETTINGS.info_log_file_path, rotation="50 MB", level="INFO", format=SETTINGS.log_format,
           enqueue=True)
logger.add(SETTINGS.error_log_file_path, rotation="50 MB", level="ERROR", format=SETTINGS.log_format,
           enqueue=True)
# logger = logger.bind(status=1)
