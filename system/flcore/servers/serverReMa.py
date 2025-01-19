# FedReMa

import time
import torch
import torch.nn.functional as F
from flcore.clients.clientReMa import clientLH
from flcore.servers.serverbase import Server
from collections import defaultdict
import numpy as np
import copy
import random
import h5py
import os

class FedReMa(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientLH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.sim_type = args.sim_type
        self.training_type = args.training_type

        self.delta = args.delta
        self.feature_size = get_first_layer_input_size(args.head)

        self.similarity = np.zeros((self.num_clients, self.num_clients))
        self.global_protos = [None for _ in range(args.num_classes)]
        self.uploaded_protos = []
        self.Budget = []
        # record neighbors (the times they are involved)
        self.neighbor = np.zeros((self.num_clients, self.num_clients))
        self.history_max_diff = np.array([])
        # is ccp
        self.is_CLP = True

        self.history_sim = np.zeros((self.num_clients, self.num_clients, self.global_rounds))

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                # 0: train normally
                # 1: with proto
                # 2: separately
                # 3: separately with proto
                if self.training_type == 0:
                    client.train()
                elif self.training_type == 1:
                    client.train_with_proto()
                elif self.training_type == 2:
                    client.train_separately()
                elif self.training_type == 3:
                    client.train_separately_with_proto()

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            self.receive_models()

            if self.sim_type == 0: 
                # 0: guassian
                self.compute_similarity_guassian()
            else:
                # 1: proto
                self.compute_similarity_proto()
            
            self.history_sim[:,:,i-1] = self.similarity

            # aggregate bodies of all clients, aggregate heads personalizedly
            self.aggregate_bases(self.uploaded_weights)
            self.aggregate_heads_diff()
            self.send_models(base=True, head=False)

            if self.is_CLP:
                print(f"\tMax Difference: {self.history_max_diff[-1]}")
            

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
    
    def compute_similarity_proto(self):
        def min_max_scale(vec):
            # 获取矩阵的行数和列数
            num_rows = len(vec)
            min_value = np.min(vec)
            max_value = np.max(vec)
            return (vec - min_value) / (max_value - min_value)
        
        def soft_prediction(logits, tem = 0.5):
            if tem == 1.0:
                # 当温度为1时，直接使用 softmax 函数进行预测
                pred_probs = F.softmax(logits, dim=-1)
            else:
                # 应用温度调节的 softmax 函数
                scaled_logits = logits / tem
                pred_probs = F.softmax(scaled_logits, dim=-1)
            return pred_probs

        for client_id in sorted(self.uploaded_ids):
            similar_matrix = np.zeros((self.num_classes, self.num_clients)) # 初始化大小为: 客户端数量 x 分类数量
            with torch.no_grad():
                for c in self.global_protos.keys():
                    proto = self.global_protos[c]
                    probs = self.clients[client_id].model.head(proto)
                    predicts = [None] * self.num_clients
                    predicts[client_id] = torch.softmax(probs, dim=0) 
                    for j in sorted(self.uploaded_ids):
                        if j==client_id:
                            similar_matrix[c, j] = 1
                        else:
                            predicts[j] = torch.softmax(self.clients[j].model.head(proto), dim=0) 
                            similar_matrix[c, j] = F.cosine_similarity(predicts[client_id], predicts[j], dim=0)
            
            # similar_vector = np.mean(min_max_scale(similar_matrix), axis=0)  # 拉一下上下界
            similar_vector = np.mean(similar_matrix, axis=0)  
            self.similarity[client_id] = similar_vector
        return 

    def compute_similarity_guassian(self):
        def min_max_scale(vec):
            # 获取矩阵的行数和列数
            num_rows = len(vec)
            min_value = np.min(vec)
            max_value = np.max(vec)
            return (vec - min_value) / (max_value - min_value)
        
        def soft_prediction(logits, tem = 0.5):
            if tem == 1.0:
                # 当温度为1时，直接使用 softmax 函数进行预测
                pred_probs = F.softmax(logits, dim=-1)
            else:
                # 应用温度调节的 softmax 函数
                scaled_logits = logits / tem
                pred_probs = F.softmax(scaled_logits, dim=-1)
            return pred_probs

        for client_id in sorted(self.uploaded_ids):
            similar_ram = np.zeros(self.num_clients) # 初始化大小为: 客户端数量 x 分类数量
            with torch.no_grad():
                rdmap = torch.rand_like(self.global_protos[0]) * 0.5 # 使用随机的高斯图像测试不同分类器的响应
                probs = self.clients[client_id].model.head(rdmap)
                predicts = [None] * self.num_clients
                predicts[client_id] = soft_prediction(probs, tem=0.5) 
                for j in sorted(self.uploaded_ids):
                    if j==client_id:
                        similar_ram[j] = 1
                    else:
                        predicts[j] = torch.softmax(self.clients[j].model.head(rdmap), dim=0) 
                        similar_ram[j] = F.cosine_similarity(predicts[client_id], predicts[j], dim=0)
        
            # similar_vector = min_max_scale(similar_ram)  # 拉一下上下界
            similar_vector = similar_ram
            self.similarity[client_id] = similar_vector
        return 
    
    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_bases = []
        self.uploaded_heads = []
        tot_samples = 0
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
                self.uploaded_bases.append(client.model.base)
                self.uploaded_heads.append(client.model.head)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        return     

# 这里这里！！！！！
    def aggregate_heads_diff(self):
        if self.is_CLP:
            max_diffs = []
            n = len(self.uploaded_ids)
            for cid in sorted(self.uploaded_ids):
                weights = [0]*self.num_clients 
                sim , sim_args , sim_diff = get_sort_grad(self.similarity[cid], n)
                max_id = np.argmax(sim_diff[:-1]) # 找到最大差值的id，选取此后的为聚合
                num_agg = len(sim_args)-1-max_id # 算一下要聚合多少个
                for id in range(1, num_agg+1):
                    weights[sim_args[-id]] = 1/num_agg
                    self.neighbor[cid, sim_args[-id]] += 1

                # 聚合
                print('(Client {}) Chosen Classifier Head'.format(cid))
                print('\t',[sim_args[-aa] for aa in range(1, num_agg+1)])
                if self.similarity[cid] is not None:
                    new_head = copy.deepcopy(self.uploaded_heads[0])
                    for param in new_head.parameters():
                        param.data.zero_()
                    for w, head in zip(weights, self.uploaded_heads):
                        for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                            server_param.data += client_param.data.clone() * w
                else:
                    new_head = self.uploaded_heads[cid]
                self.clients[cid].set_head(new_head)
                
                # 维护
                max_diffs.append(np.max(sim_diff[:-1]))
            self.history_max_diff = np.append(self.history_max_diff, (np.average(max_diffs)))
            self.check_critical()
        else:
            for cid in sorted(self.uploaded_ids):
                weights = [prob/sum(self.neighbor[cid]) for prob in self.neighbor[cid]]
                print('(Client {}) Classifier Weights'.format(cid))
                print('\t',[round(weights[id], 3) for id in range(len(weights))])
                if self.similarity[cid] is not None:
                    new_head = copy.deepcopy(self.uploaded_heads[0])
                    for param in new_head.parameters():
                        param.data.zero_()
                    for w, head in zip(weights, self.uploaded_heads):
                        for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                            server_param.data += client_param.data.clone() * w
                else:
                    new_head = self.uploaded_heads[cid]
                self.clients[cid].set_head(new_head)
        return
    
    def aggregate_bases_diff(self):
        if self.is_CLP:
            max_diffs = []
            n = len(self.uploaded_ids)
            for cid in sorted(self.uploaded_ids):
                weights = [0]*self.num_clients 
                sim , sim_args , sim_diff = get_sort_grad(self.similarity[cid], n)
                max_id = np.argmax(sim_diff[:-1]) # 找到最大差值的id，选取此后的为聚合
                num_agg = len(sim_args)-1-max_id # 算一下要聚合多少个
                for id in range(1, num_agg+1):
                    weights[sim_args[-id]] = 1/num_agg
                    self.neighbor[cid, sim_args[-id]] += 1

                # 聚合
                print('(Client {}) Chosen Body'.format(cid))
                print('\t',[sim_args[-aa] for aa in range(1, num_agg+1)])
                if self.similarity[cid] is not None:
                    new_base = copy.deepcopy(self.uploaded_bases[0])
                    for param in new_base.parameters():
                        param.data.zero_()
                    for w, base in zip(weights, self.uploaded_bases):
                        for server_param, client_param in zip(new_base.parameters(), base.parameters()):
                            server_param.data += client_param.data.clone() * w   
                else:
                    new_base = self.uploaded_bases[cid]
                self.clients[cid].set_base(new_base)
                
                # 维护
                max_diffs.append(np.max(sim_diff[:-1]))
            self.history_max_diff = np.append(self.history_max_diff, (np.average(max_diffs)))
            self.check_critical()
        else:
            for cid in sorted(self.uploaded_ids):
                weights = [prob/sum(self.neighbor[cid]) for prob in self.neighbor[cid]]
                print('(Client {}) Body Weights'.format(cid))
                print('\t',[round(weights[id], 3) for id in range(len(weights))])
                if self.similarity[cid] is not None:
                    new_base = copy.deepcopy(self.uploaded_bases[0])
                    for param in new_base.parameters():
                        param.data.zero_()
                    for w, base in zip(weights, self.uploaded_bases):
                        for server_param, client_param in zip(new_base.parameters(), base.parameters()):
                            server_param.data += client_param.data.clone() * w   
                else:
                    new_base = self.uploaded_bases[cid]
                self.clients[cid].set_base(new_base)
        return
    
    def aggregate_all_diff(self):
        if self.is_CLP:
            max_diffs = []
            n = len(self.uploaded_ids)
            for cid in sorted(self.uploaded_ids):
                weights = [0]*self.num_clients 
                sim , sim_args , sim_diff = get_sort_grad(self.similarity[cid], n)
                max_id = np.argmax(sim_diff[:-1]) # 找到最大差值的id，选取此后的为聚合
                num_agg = len(sim_args)-1-max_id # 算一下要聚合多少个
                for id in range(1, num_agg+1):
                    weights[sim_args[-id]] = 1/num_agg
                    self.neighbor[cid, sim_args[-id]] += 1

                # 聚合
                print('(Client {}) Chosen Weight'.format(cid))
                print('\t',[sim_args[-aa] for aa in range(1, num_agg+1)])
                if self.similarity[cid] is not None:
                    new_head = copy.deepcopy(self.uploaded_heads[0])
                    new_base = copy.deepcopy(self.uploaded_bases[0])
                    for param in new_head.parameters():
                        param.data.zero_()
                    for param in new_base.parameters():
                        param.data.zero_()  
                    for w, head in zip(weights, self.uploaded_heads):
                        for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                            server_param.data += client_param.data.clone() * w
                    for w, base in zip(weights, self.uploaded_bases):
                        for server_param, client_param in zip(new_base.parameters(), base.parameters()):
                            server_param.data += client_param.data.clone() * w   
                else:
                    new_base = self.uploaded_bases[cid]
                    new_head = self.uploaded_heads[cid]
                self.clients[cid].set_base(new_base)
                self.clients[cid].set_head(new_head)
                
                # 维护
                max_diffs.append(np.max(sim_diff[:-1]))
            self.history_max_diff = np.append(self.history_max_diff, (np.average(max_diffs)))
            self.check_critical()
        else:
            for cid in sorted(self.uploaded_ids):
                weights = [prob/sum(self.neighbor[cid]) for prob in self.neighbor[cid]]
                print('(Client {}) Weights'.format(cid))
                print('\t',[round(weights[id], 3) for id in range(len(weights))])
                if self.similarity[cid] is not None:
                    new_head = copy.deepcopy(self.uploaded_heads[0])
                    new_base = copy.deepcopy(self.uploaded_bases[0])
                    for param in new_head.parameters():
                        param.data.zero_()
                    for param in new_base.parameters():
                        param.data.zero_()  
                    for w, head in zip(weights, self.uploaded_heads):
                        for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                            server_param.data += client_param.data.clone() * w
                    for w, base in zip(weights, self.uploaded_bases):
                        for server_param, client_param in zip(new_base.parameters(), base.parameters()):
                            server_param.data += client_param.data.clone() * w                    
                else:
                    new_base = self.uploaded_bases[cid]
                    new_head = self.uploaded_bases[cid]
                self.clients[cid].set_base(new_base)
                self.clients[cid].set_head(new_head)
        return


    def aggregate_heads(self, weights):
        self.global_head = copy.deepcopy(self.uploaded_heads[0])
        for param in self.global_head.parameters():
            param.data.zero_()                    
        for w, head in zip(weights, self.uploaded_heads):
            for server_param, client_param in zip(self.global_head.parameters(), head.parameters()):
                server_param.data += client_param.data.clone() * w

    # 需要重写
    def aggregate_bases(self, weights, is_default = False):
        self.global_base = copy.deepcopy(self.uploaded_bases[0])
        for param in self.global_base.parameters():
            param.data.zero_()            
        for w, base in zip(weights, self.uploaded_bases):
            for server_param, client_param in zip(self.global_base.parameters(), base.parameters()):
                server_param.data += client_param.data.clone() * w

    def send_models(self, base=True, head=True):
        for client in self.clients:
            start_time = time.time()
            if base:
                client.set_base(self.global_base)
            if head:
                client.set_head(self.global_head)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # def check_critical2(self, thre1=0, thre2=0.3):
    #     maxi = np.argmax(self.history_max_diff)
    #     maxv = np.max(self.history_max_diff)
    #     now = len(self.history_max_diff)-1
    #     if maxi == now:
    #         return
    #     else:
    #         window = min(now-maxi+1, self.window)
    #         judge1 = (self.history_max_diff[now]-self.history_max_diff[now-window+1])/(window-1) # 窗口斜率
    #         judge2 = (maxv-self.history_max_diff[now])/maxv
    #     if judge1 >= thre1 and judge2 > thre2:
    #         self.is_CLP = False
    #     return
    
    # def check_critical1(self, ratio=0.50, thre=0.50):
    #     maxi = np.argmax(self.history_max_diff)
    #     maxv = np.max(self.history_max_diff)
    #     now = len(self.history_max_diff)-1
    #     if maxi == now:
    #         return
    #     if self.history_max_diff[now]/maxv < ratio or self.history_max_diff[now]<thre:
    #         self.is_CLP = False
    #     return

    def check_critical(self):
        delta = self.delta
        maxi = np.argmax(self.history_max_diff)
        maxv = np.max(self.history_max_diff)
        now = len(self.history_max_diff)-1
        if maxi == now:
            return
        if  self.history_max_diff[now]/maxv < delta:
            self.is_CLP = False
        return
    
    def save_results(self, savepath="../results/"):
        algo = self.dataset + "_" + self.algorithm
        result_path = savepath
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('history_max_diff', data=self.history_max_diff)
                hf.create_dataset('history_sim', data=self.history_sim)

def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label

def get_sort_grad(sim, n):
    sorted_values = np.sort(sim)[-n:]
    sorted_args = np.argsort(sim)[-n:]
    sim_diff = np.diff(sorted_values)
    return sorted_values, sorted_args, sim_diff

def get_first_layer_input_size(module):
    if isinstance(module, torch.nn.Linear):
        return module.in_features
    elif isinstance(module, torch.nn.Sequential):
        return get_first_layer_input_size(module[0])
    else:
        raise ValueError("Unsupported network structure.")