"""
保存模型，最佳模型参数，打印模型最佳指标

"""


import numpy as np
import os
import torch

#输出写入器，负责保存模型
class OutputSave( object):
    def __init__(self,model,args,optimizer):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.best_top1 = 0.0
        self.best_top5 = 0.0



    #保存checkpoint
    def save_checkpoint(self,epoch):
        root = self.args.save_path
        os.makedirs(root, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'best_top1': self.best_top1,
            'best_top5': self.best_top5,
        }


        torch.save(checkpoint, os.path.join(root, 'checkpoint.pth.tar'))





    #更新最佳模型
    def update_best(self,top1,top5,epoch):
        if top1 > self.best_top1:
            self.best_top1 = top1
        if top5 > self.best_top5:
            self.best_top5 = top5
            root = self.args.save_path
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_top1': self.best_top1,
                'best_top5': self.best_top5,
            }
            torch.save(checkpoint, os.path.join(root, 'best_model.pth.tar'))
            print(f"保存最佳模型,准确率 -> top1:{self.best_top1:.4f}%  top5:{self.best_top5:.4f}%")