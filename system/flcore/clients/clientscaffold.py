import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import SCAFFOLDOptimizer


class clientSCAFFOLD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_c, self.client_c)

        # self.model.cpu()
        self.num_batches = len(trainloader)
        self.update_yc(max_local_epochs)
        # self.delta_c, self.delta_y = self.delta_yc(max_local_epochs)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
            
        
    def set_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = model

    def update_yc(self, max_local_epochs=None):
        """更新client的c_i, 使用的是第二种更新方式
        
            SCAFFOLD论文中并未提及c_i的更新需要涉及batch数量(self.num_batches)
            代码实现中额外加上这一参数可能是作者在工程上的调整(可能是为了归一化或平滑参数更新, 使得不同batch_size或epoch更新幅度保持一致)
        """
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1/self.num_batches/max_local_epochs/self.learning_rate * (x - yi)

    def delta_yc(self, max_local_epochs=None):
        """计算更新后的本地client模型y_i和全局模型x之间的差值, 以及更新后的c_i^+和原c_i的差值, 以便更新全局模型x和全局控制变量c"""
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1/self.num_batches/max_local_epochs/self.learning_rate * (x - yi))

        return delta_y, delta_c

