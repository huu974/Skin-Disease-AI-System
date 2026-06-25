""""
作用：一阶优化器工厂模块，根据配置创建优化器
"""

from utils.optimizer_factory import build_optimizer


#创建一阶优化器
def SFO(model,args):
    return build_optimizer(model.parameters(), args)
















