# FedReMa

[![arXiv](https://img.shields.io/badge/arXiv-2411.01825-b31b1b.svg)](https://arxiv.org/abs/2411.01825)

We are very grateful to **[PFLlib](https://github.com/TsingZ0/PFLlib)** for providing an excellent personalized federated learning framework. This has brought a lot of convenience to our experiment. Therefore, I will refactor FedReMa's code based on the PFLLIB framework and open source it in this Repository.

## Installation
The code requirements are consistent with PFLlib
### Environments
Install CUDA.
Install conda latest and activate conda.
```
conda env create -f env.yaml # Downgrade torch via pip if needed to match the CUDA version
```
### Clone
```
git clone https://github.com/liangh68/FedReMa.git
```

## Usage
On the basis of PFLlib, we have modified and rewritten the code for generating the dataset. And only retained the code related to our algorithm, removing irrelevant redundant parts.

### Examples for CIFAR10

Before generating the dataset, you must download the rawdata then organize them like this:
```
dataset/
    |—— cifar10/
        |—— rawdata/ # download there
            |—— cifar-10-batches-py/
            |—— cifar-10-python.tar.gz
    |—— fmnist/
    |—— mnist/
```
Then runing `generate_dataset.py` by:
```
cd ./dataset
python generate_dataset.py --dataset cifar10 -K 20 -npc 500 -tr 0.8
```
It means we generate a CIFAR-10 dataset and split it to 20 clients. Each client has 500 samples while only 500*0.8=400 samples are for training.

### Run
Enter the repository file:
```
cd FedReMa
```
Run the main.py:
```
python main.py -algo FedReMa -data cifar10 -dtp path/to/dataset
```
or run it nohuply:
```
nohup python -u main.py -algo FedReMa -go test -data cifar10 -gr 500 -dtp path/to/dataset > FedReMa.out 2>&1 &
```
