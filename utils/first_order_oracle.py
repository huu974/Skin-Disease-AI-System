""""
作用：一阶优化器工厂模块，根据配置创建优化器
"""

import torch
from utils.optimizer_Adam import CustomAdam


#创建一阶优化器
def SFO(model,args):
    if args.optimizer == 'Adam':
        return CustomAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
















