# -*- coding: UTF-8 -*-
"""
@Time : 09/03/2025 11:20
@Author : xiaoguangliang
@File : global_setting.py
@Project : faice
"""
import os
import json

# ********************************************* PATH SETTING ********************************************* #

# 项目基础路径
BASE_DIR = os.path.dirname(os.path.dirname(__file__))


# ********************************************* Env setting ********************************************* #

class Config(object):
    # *************************************** log path setting *************************************** #

    # log path
    LOG_DIR = os.getenv('LOG_DIR', BASE_DIR + '/logs/')
    debug_log_file_path = os.path.join(LOG_DIR, "debug.log")
    info_log_file_path = os.path.join(LOG_DIR, "info.log")
    error_log_file_path = os.path.join(LOG_DIR, "error.log")

    # log form
    log_format = "{time:YYYY-MM-DD HH:mm:sss} | {message}"

    # *************************************** sentry setting *************************************** #

    SENTRY_URL = 'https://43c4683ceda6404549f18ee0aaa0f642@o421658.ingest.us.sentry.io/4509028258217984'


class Development(Config):
    pass


class Production(Config):
    pass


# Set the default settings
settings = {
    "default": Config,
    "development": Development,
    "production": Production
}

# Get the environment configuration key from the environment variable
SETTINGS = settings[os.getenv('ENV', 'default')]

if __name__ == '__main__':
    print(settings['default'].debug_log_file_path)
    print(settings['development'].debug_log_file_path)
    print(settings['production'].debug_log_file_path)
