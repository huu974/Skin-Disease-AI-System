"""
写入tensorboard，文件，终端的写入器
"""

import sys, os
import torch
from utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import time


# 初始化日志记录器和TensorBoard写入器
def init_writer(args):
    datatime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # 创建日志目录和文件路径
    logdir = f'./run/{datatime}'
    logfile = os.path.join(logdir, 'train.logs')

    #设置日志输出到文件
    sys.stdout = Logger(logfile, args.logterminal)
    
    #打印所有配置参数
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    #创建run目录用于保存TensorBoard数据
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    #创建TensorBoard写入器
    writer = SummaryWriter(log_dir=logdir)

    # 打印分布式训练信息
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("使用 CPU")

    return writer