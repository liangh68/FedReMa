# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import random

from flcore.servers.serverReMa import FedReMa

from flcore.trainmodel.models import *
from flcore.trainmodel.alexnet import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

my_seed = 1
torch.manual_seed(my_seed)
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "cnn": # non-convex
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            else:
                raise NotImplementedError

        elif model_str == "alexnet":
            if "cifar10" in args.dataset:
                args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedReMa":
            if model_str == "alexnet":
                args.head = copy.deepcopy(nn.Sequential(args.model.classifier,args.model.fc))
                args.model.classifier = nn.Identity()
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
            elif model_str == "cnn": 
                args.head = nn.Sequential(args.model.fc1,args.model.fc)
                args.model.fc1 = nn.Identity()
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
            else:
                raise NotImplementedError


        if args.algorithm == "FedReMa":
            server = FedReMa(args, i)
            
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # 
    parser.add_argument('-go', "--goal", type=str,          default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str,       default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str,    default="0")

    parser.add_argument('-data', "--dataset", type=str,     default="cifar10") 
    parser.add_argument('-nb', "--num_classes", type=int,   default=10)
    parser.add_argument('-dtp', "--datapath", type=str,     default="dataset/") # 
    # 
    parser.add_argument('-m', "--model", type=str,          default="alexnet", choices=["alexnet", "cnn"]) 
    parser.add_argument('-lbs', "--batch_size", type=int,   default=100) # 10
    parser.add_argument('-lr', "--local_learning_rate", type=float,         default=0.01,
                        help="Local learning rate") # 0.005
    parser.add_argument('-ld', "--learning_rate_decay", type=bool,          default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float,  default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int,                 default=500) # 2000
    parser.add_argument('-ls', "--local_epochs", type=int,                  default=5, # 1
                        help="Multiple update steps in one local epoch.")
    
    # FedReMa
    parser.add_argument('-hln', "--head_layers_num", type=int, default=3)
    parser.add_argument('-lam', "--lamda", type=float,  default=1.0,
                        help="Regularization weight")
    parser.add_argument('-del', "--delta", type=float,  default=0.5,
                        help="Threshold (delta)")
    parser.add_argument('-smo', "--sgd_momentum", type=float, default=0.9) # 0.1
    
    parser.add_argument('-sim', "--sim_type", type=int, default=0) #
    parser.add_argument('-tt', "--training_type", type=int, default=0) #
    parser.add_argument('-up', "--use_proto", type=bool, default=False)
    # Options for simlarity computing：
    # 0: guassian
    # 1: proto
    # Options for training：
    # 0: train normally
    # 1: with proto
    # 2: separately
    # 3: separately with proto

    # (Not all of them are used)
    parser.add_argument('-algo', "--algorithm", type=str,                   default="FedReMa")

    parser.add_argument('-jr', "--join_ratio", type=float,                  default=1, # 1
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool,           default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int,                   default=20, 
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    else:
        torch.cuda.set_device(int(args.device_id))


    print("=" * 50)
    print("Dataset: {}".format(args.datapath))
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    print("SGD momentum: {}".format(args.sgd_momentum))
    
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    if args.algorithm == "FedReMa":
        print("Training type: {}".format(args.training_type))
        print("Threshold: {}".format(args.delta))
    print("=" * 50)

    run(args)

