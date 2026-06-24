"""
自定义Adam优化器
"""

import torch
import math

class CustomAdam(torch.optim.Adam):
    #装饰器禁用梯度计算
    @torch.no_grad()
    #执行一次优化步骤
    def step(self):
        loss = None

        #遍历参数组        self.param_groups 优化器的参数组
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                #参数的状态字典  -> 保存参数的更新历史，不同参数相互独立
                state = self.state[p]

                #状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    #一阶矩估计 ：平滑更新，减少震荡     参2：控制张量的内存格式  -> 确保缓冲区与参数 p 的内存格式完全一致，避免后续操作因格式不匹配而产生额外转换。
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    #二阶矩估计: 对每个参数进行动态调整，提升模型的鲁棒性
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg,exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                #偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                #权重衰减
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  #等价于 exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) #等价于 exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

                #权重衰减
                if group['weight_decay'] != 0:
                    #添加惩罚系数
                    grad = grad.add(p, alpha=group['weight_decay'])
                #
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                #偏差矫正  lr_t = lr / (1 - β1^t)
                step_size = group['lr'] / bias_correction1


                #参数更新
                p.addcdiv_(exp_avg, denom, value=-step_size)
                # p = p - step_size * (exp_avg / denom)
        return loss













