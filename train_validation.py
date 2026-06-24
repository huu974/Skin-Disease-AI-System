import time
import numpy as np
from tqdm import tqdm

import torch
from torch import autocast

from utils.dataset import get_train_dataloader
from utils.dataset import get_val_dataloader
from utils.dataset import mixup_cutmix_data
from utils.arguments import parse
import torch.backends.cudnn as cudnn    #CUDA加速库
from utils.lr_policy import LR
from utils.first_order_oracle import CustomAdam

class tra_val(object):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 scaler,
                 args,
                 train_loader,
                 val_loader,
                 writer,
                 ):

        #1.存储参数
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        #存储每一个batch的loss
        self.loss_vector = list()
        #存储每一个batch的准确率
        self.acc_vector = list()
        #当前epoch的累计 loss
        self.train_loss = 0
        #所有epoch的loss
        self.total_loss = 0
        self.step = 0

        #使用列表推导式创建三个空列表  等价于# self.TrainingLoss = [] self.TrainingTop1 = [] self.TrainingTop5 = []
        self.TrainingLoss, self.TrainingTop1, self.TrainingTop5 = ([] for i in range(3))
        self.TestLoss, self.TestTop1, self.TestTop5 = ([] for i in range(3))
        # 学习率历史，初始为初始学习率
        self.Learning_Rate = [self.args.lr]
        #学习率策略（余弦退火 + Warmup）
        steps_per_epoch = len(train_loader) if train_loader else (len(val_loader) if val_loader else 100)
        self.lr_policy = LR(base_lr=self.args.lr,warmup_epoch=5,epochs=self.args.epochs,steps_per_epoch=steps_per_epoch)

    #模型训练
    def train(self, epoch):
        #1.初始化
        self.epoch = epoch


        #创建指标计量器
        self.losses_tr = AverageMeter()             #训练损失
        self.top1_tr = AverageMeter()               #Top-1准确率
        self.top5_tr = AverageMeter()               #Top-5准确率


        #2.切换模型为训练模式
        self.model.train()

        #3.计算训练开始时间
        self.end_tr = time.time()

        #4.训练开始
        for i, (input, target) in enumerate(tqdm(self.train_loader, desc=f'训练 Epoch {self.epoch+1}')):
            #5.计算应用学习率并应用到优化器更新
            self.lr = self.lr_policy.apply_lr(epoch, i)
            self.assign_learning_rate(self.lr)

            #6.确定内存格式和半精度
            if self.args.channels_last:
                input = input.to(memory_format=torch.channels_last)
            input = input.to(device='cuda')
            target = target.to(device='cuda')
            # 7.随机应用MixUp或CutMix（50%概率）
            use_mixup = False
            lam = 1.0
            target_a = target
            target_b = target
            if np.random.rand() < 0.5:
                input, target_a, target_b, lam = mixup_cutmix_data(input, target)
                use_mixup = True

            #8.前向传播(混合精度)
            with autocast(device_type='cuda'):
                output = self.model(input)
                if use_mixup:
                    loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(output, target_b)
                else:
                    loss = self.criterion(output, target)

                #9.反向传播
                self.optimizer.zero_grad()
                #10.混合精度，缩放损失，防止梯度下溢
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()


                #11.更新参数(混合精度)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)        #根据scale后的梯度跟新参数
                    self.scaler.update()                    #调整scale因子
                else:
                    self.optimizer.step()
                self.step += 1  # 累计step数

                #12.计算准确率
                self.prec1_tr,self.prec5_tr = self.accuracy(output,target,topk=(1,5))

                #13.更新参数量
                self.losses_tr.update(loss.item(),input.size(0))
                self.top1_tr.update(self.prec1_tr.item(),input.size(0))
                self.top5_tr.update(self.prec5_tr.item(),input.size(0))
                self.loss = loss

                #14.写入TensorBoard
                self.write_net_values(train=True)



        print(f'训练完成 | Top1: {self.top1_tr.avg:.2f}% | Top5: {self.top5_tr.avg:.2f}%  | loss: {self.losses_tr.avg:.4f}')



    #模型验证
    def validation(self, epoch):
        """
        epoch:当前epoch
        report：是否打印报告
        """
        #1.初始化
        self.epoch = epoch
        self.batch_time_ts = AverageMeter()
        self.losses_ts = AverageMeter()
        self.top1_ts = AverageMeter()
        self.top5_ts = AverageMeter()

        #2.切换模型为验证模式
        self.model.eval()

        #3.计算验证开始时间
        self.end_ts = time.time()

        #4.验证开始
        for i, (input, target) in enumerate(tqdm(self.val_loader, desc=f'验证 Epoch {self.epoch+1}')):
            #确定内存格式和半精度
            if self.args.channels_last:
                input = input.to(memory_format=torch.channels_last)
            input = input.to(device='cuda')
            target = target.to(device='cuda')
            #7.前向传播
            with torch.no_grad():
                output = self.model(input)
                loss = self.criterion(output, target)
            #8.计算准确率
            self.prec1_ts,self.prec5_ts = self.accuracy(output,target,topk=(1,5))
            self.loss_ts = loss

            #9.更新参数
            self.losses_ts.update(self.loss_ts.item(),input.size(0))
            self.top1_ts.update(self.prec1_ts.item(),input.size(0))
            self.top5_ts.update(self.prec5_ts.item(),input.size(0))

            #10.更新时间
            self.batch_time_ts.update((time.time() - self.end_ts))
            self.end_ts = time.time()


            # 11.记录验证阶段的指标
            self.write_net_values(train=False)

        print(f'验证完成 | Top1: {self.top1_ts.avg:.2f}% | Top5: {self.top5_ts.avg:.2f}% | loss: {self.losses_ts.avg:.4f}')
        return self.losses_ts.avg, self.top1_ts.avg, self.top5_ts.avg






    #写入网格指标到TensorBoard
    def write_net_values(self,train):
        if self.writer is None:
            return
        if train:
            self.writer.add_scalar('Loss/Training',self.loss.item(),self.step)
            self.writer.add_scalar('Top1/Training',self.prec1_tr.item(),self.step)
            self.writer.add_scalar('Optim/lr', self.lr, self.step)
            self.writer.add_scalar('Top5/Training', self.prec5_tr.data.item(), self.step)

            self.TrainingLoss.append(self.loss.item())
            self.TrainingTop1.append(self.prec1_tr.item())
            self.TrainingTop5.append(self.prec5_tr.item())

        #验证时
        else:
            self.writer.add_scalar('Loss/Test', self.losses_ts.avg, self.epoch + 1)
            self.writer.add_scalar('Top1/Test', self.top1_ts.avg, self.epoch + 1)
            self.writer.add_scalar('Top5/Test', self.top5_ts.avg, self.epoch + 1)

            self.TestLoss.append(self.losses_ts.avg)
            self.TestTop1.append(self.top1_ts.avg)
            self.TestTop5.append(self.top5_ts.avg)



    #更新优化器的学习率
    def assign_learning_rate(self,new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr



    #计算topk准确率
    def accuracy(self,output,target,topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _,pred = output.topk(maxk,1,True,True)  #获取概率最高的前k的类别
        pred = pred.t()                         #转置               [maxk, batch_size]
        correct = pred.eq(target.view(1,-1).expand_as(pred))        #判断预测类别是否与真实值一致

        res = []
        # 计算top-k准确率
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))

        return res





# 指标计量器
class AverageMeter(object):
    def __init__(self):
        self.reset()    #在每一个epoch中调用，清空所有值

    def reset(self):
        self.val = 0        #当前值
        self.avg = 0        #平均值
        self.sum = 0        #累计和
        self.count = 0      #样本数

    #更新计量器
    def update(self,val,n=1):
        """
        val:新值
        n:样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
