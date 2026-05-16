"""
logger.py

这个文件用于定义日志工具。

作用：
1. 在终端打印训练过程信息
2. 将训练信息写入日志文件
3. 方便后续查看实验过程和排查问题
"""

import os
from datetime import datetime


def get_time_str():
    """
    生成当前时间字符串，用于日志文件命名。
    """

    return datetime.now().strftime("[%m-%d]-[%H-%M]")


def make_log_file(log_dir):
    """
    创建日志文件路径。

    参数：
        log_dir: 日志文件夹路径

    返回：
        log_path: 日志文件完整路径
    """

    os.makedirs(log_dir, exist_ok=True)
    log_name = get_time_str() + "-log.txt"
    log_path = os.path.join(log_dir, log_name)
    return log_path


def write_log(log_path, text):
    """
    将一段文本追加写入日志文件。
    """

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def print_and_log(log_path, text):
    """
    同时在终端打印并写入日志文件。
    """

    print(text)
    write_log(log_path, text)