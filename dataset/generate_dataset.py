import numpy as np
import random, os
import argparse
from generator import DatasetGenerator

random.seed(1)
np.random.seed(1)

## usage:
# python generate_dataset.py --dataset cifar10 -K 20 -npc 500 -tr 0.8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=["cifar10", "fmnist", "mnist"])
    parser.add_argument('-K', "--num_clients", type=int, default=20)
    parser.add_argument('-C', "--num_classes", type=int, default=10)
    parser.add_argument('-npc', "--num_per_client", type=int, default=500)
    parser.add_argument('-tr', "--train_ratio", type=float, default=0.8)

    parser.add_argument("--partition_method", type=str, default='by_s')

    # for 'by_s' method
    parser.add_argument("--s", type=float, default=0.2,
                        help="If partition method is 'by_s', 's' will be used.")
    args = parser.parse_args()
    args.root_dir = os.path.dirname(__file__)


    data_gen = DatasetGenerator(args)
    data_gen.generate()
