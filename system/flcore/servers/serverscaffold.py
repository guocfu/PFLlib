import copy
import random
import time
import torch
from flcore.clients.clientscaffold import clientSCAFFOLD
from flcore.servers.serverbase import Server
from threading import Thread


class SCAFFOLD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientSCAFFOLD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.server_learning_rate = args.server_learning_rate
        self.global_c = []
        for param in self.global_model.parameters():
            self.global_c.append(torch.zeros_like(param))


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientSCAFFOLD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def send_models(self):
        """重写send_models方法, 为所有的client更新全局模型以及通信成本
            
            因为clientSCAFFOLD的set_parameters方法所需参数有所改变, 不能用Server基类的send_models方法"""
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model, self.global_c)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """重写receive_models方法, server接受符合条件的client信息
        
            server不用接受client的模型信息, 因为更新全局模型x和全局控制变量所需的delta_y和delta_c可以由clientSCAFFOLD.update_yc()获得
        """
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        # self.delta_ys = []
        # self.delta_cs = []
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                # self.delta_ys.append(client.delta_y)
                # self.delta_cs.append(client.delta_c)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """重写aggregate_parameters()方法, 更新全局模型x以及全局控制变量c
        """
        # # original version
        # for dy, dc in zip(self.delta_ys, self.delta_cs):
        #     for server_param, client_param in zip(self.global_model.parameters(), dy):
        #         server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate
        #     for server_param, client_param in zip(self.global_c, dc):
        #         server_param.data += client_param.data.clone() / self.num_clients
        
        # save GPU memory
        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        for cid in self.uploaded_ids:
            dy, dc = self.clients[cid].delta_yc()
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate # .clone()是一种安全措施, 确保server聚合时用的是参数的副本, 不会影响客户端原始数据, 这样做更稳妥
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() / self.num_clients
        self.global_model = global_model
        self.global_c = global_c

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        """与serverbase中fine_tuning_new_clients()方法一模一样, 有必要重写吗?"""
        for client in self.new_clients:
            client.set_parameters(self.global_model, self.global_c)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()