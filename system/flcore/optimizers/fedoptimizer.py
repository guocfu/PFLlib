import torch
from torch.optim import Optimizer


class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        """进行一次参数更新, 更新公式为: 
        y_i ← y_i − \eta _l (g_i(y_i) − c_i + c) 

        args:
            server_cs: server的控制变量c
            client_cs: client的本地控制变量c_i
            二者与模型参数同维, 逐元素参与运算
        """
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr) 
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)


class PerturbedGradientDescent(Optimizer):
    """扰动梯度下降优化器(FedProx使用)
    
        - 该优化器在普通梯度下降的基础上，增加了一个“扰动”项：mu * (p.data - g.data)，即本地参数与全局参数的差异，乘以系数mu, 该项用于参数更新, 是已经求导过的式子
        - 这样做的目的是在联邦学习中，鼓励本地模型参数向全局模型靠拢，防止本地模型偏离过远。
        - @torch.no_grad() 保证参数更新时不会影响梯度计算图，节省内存并避免梯度污染。
    """
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu) # 将学习率和mu封装成字典
        super().__init__(params, default) # 调用父类Optimizer的构造函数，初始化参数组, 包含self.default=self.default

    @torch.no_grad()  # 装饰器，表示该方法在执行时不需要计算梯度（不会影响autograd的计算图）
    def step(self, global_params, device): # 进行一次参数更新
        for group in self.param_groups:  # 遍历所有参数组（通常只有一组）
            for p, g in zip(group['params'], global_params):  # 同时遍历本地参数p和全局参数g
                g = g.to(device)  # 将全局参数g移动到指定设备上
                d_p = p.grad.data + group['mu'] * (p.data - g.data)  # 计算更新量：本地梯度 + mu*(本地参数-全局参数)
                p.data.add_(d_p, alpha=-group['lr'])   # 用学习率lr对参数进行梯度下降更新（负方向）
