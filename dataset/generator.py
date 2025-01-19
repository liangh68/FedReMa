import numpy as np
import os, glob
import json
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class DatasetGenerator:
    def __init__(self, args):
        self.dataset = args.dataset
        self.dir_path = os.path.join(args.root_dir, args.dataset)
        
        self.config_path = os.path.join(self.dir_path, "config.json")
        self.train_path = os.path.join(self.dir_path, "train")
        self.test_path = os.path.join(self.dir_path, "test")

        self.num_clients = args.num_clients
        self.num_classes = args.num_classes
        self.num_per_client = args.num_per_client
        self.train_ratio = args.train_ratio

        self.partition_method = args.partition_method
        self.s = args.s # When 's' gets higher, the dataset is more noniid.

    def _check_exist(self):
        # check existing dataset
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            if config.get('num_clients') == self.num_clients and \
                config.get('num_classes') == self.num_classes and \
                config.get('num_per_client') == self.num_per_client and \
                config.get('num_per_client') == self.num_per_client and \
                config.get('dataset') == self.dataset and \
                config.get('train_ratio') == self.train_ratio:
                print("\nDataset already generated.\n")
                return True
        dir_path = os.path.dirname(self.train_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        dir_path = os.path.dirname(self.test_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return False

    def _delete_all_npz(self):
        print("Deleting the existing dataset...")
        for path in [self.train_path, self.test_path]:
            npz_files = glob.glob(os.path.join(path, "*.npz"))
            for npz_file in npz_files:
                try:
                    os.remove(npz_file)
                except Exception as e:
                    print(f"Failed to delete {npz_file}: {e}")
        return

    # Allocate data to users
    def generate(self):
        dir_path = self.dir_path
        dataset = self.dataset
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if self._check_exist():
            print("Overwrite?(y/n)")
            yesorno = input()
            if yesorno == "y":
                pass
            elif yesorno == "n":
                return
            else: 
                raise RuntimeError("Please input 'y' or 'n'.")
        
        self._delete_all_npz()

        if dataset == "cifar10":
            trainset, testset = self._get_cifar10()
        elif dataset == "mnist":
            trainset, testset = self._get_mnist()
        elif dataset == "fmnist":
            trainset, testset = self._get_fmnist()
        else:
            raise NotImplementedError
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset.data), shuffle=False)
        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data
        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        X, y, statistic = self._separate_data(data=(dataset_image, dataset_label))
        train_data, test_data = self._split_data(images=X, labels=y)
        self._save_file(train_data, test_data, statistic)

    def _get_cifar10(self):
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), 
                                                            (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(self.dir_path, "rawdata"),
                                                train=True, 
                                                download=False, 
                                                transform=transform)
        testset = torchvision.datasets.CIFAR10(root=os.path.join(self.dir_path, "rawdata"), 
                                                train=False, 
                                                download=False, 
                                                transform=transform)
        return trainset, testset

    def _get_mnist(self):
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize([0.5], [0.5])])
        trainset = torchvision.datasets.MNIST(root=os.path.join(self.dir_path, "rawdata"), 
                                                train=True, 
                                                download=True, 
                                                transform=transform)
        testset = torchvision.datasets.MNIST(root=os.path.join(self.dir_path, "rawdata"), 
                                                train=False, 
                                                download=True, 
                                                transform=transform)
        return trainset, testset
        
    def _get_fmnist(self):
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize([0.5], [0.5])])
        trainset = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir_path, "rawdata"), 
                                                        train=True, 
                                                        download=True, 
                                                        transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=os.path.join(self.dir_path, "rawdata"), 
                                                    train=False, 
                                                    download=True, 
                                                    transform=transform)
        return trainset, testset

    def _separate_data(self, **kwargs):
        num_clients = self.num_clients
        num_classes = self.num_classes
        num_per_client = self.num_per_client
        partition = self.partition_method
        data = kwargs.get("data")
        dataset_image, dataset_label = data 

        num_all_data = len(dataset_label)

        X = [[] for _ in range(num_clients)] # images
        y = [[] for _ in range(num_clients)] # labels
        statistic = [[] for _ in range(num_clients)]
        dataidx_map = {}

        if partition == "by_s": 
            s = self.s
            label_group = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]
            label_group_num = len(label_group)

            # divide the first dataset
            num_iid = int(num_per_client*s) 
            num_noniid = num_per_client - num_iid 

            dataidx_map = {i: np.array([]) for i in range(num_clients)}
            num_per_label_total = int(num_all_data/num_classes)

            idxs = np.arange(num_all_data) # idxs
            idxs_labels = np.vstack((idxs, dataset_label))
            idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
            idxs = idxs_labels[0, :]

            # label available
            label_list = [i for i in range(num_classes)]
            # number of imgs has allocated per label
            label_used = [2000 for _ in range(num_classes)]
            num_iid_per_label = int(num_iid/num_classes)
            num_iid_per_label_last = num_iid - (num_classes-1)*num_iid_per_label

            for client in range(num_clients):

                # allocate iid idxs
                label_count = 0
                for label in label_list:
                    label_count += 1
                    num_iid_this_label = num_iid_per_label
                    start = label * num_per_label_total + label_used[label]
                    # allocate last samples of this label
                    if label_count == num_classes:
                        num_iid_this_label = num_iid_per_label_last
                    # this label is used out  
                    if (label_used[label] + num_iid_this_label) > num_per_label_total:
                        start = label * num_per_label_total
                        label_used[label] = 0
                    dataidx_map[client] = np.concatenate((dataidx_map[client], idxs[start:start+num_iid_this_label]), 
                                                    axis=0)
                    label_used[label] = label_used[label] + num_iid_this_label

                # allocate noniid idxs
                noniid_label_list = label_group[int(client*label_group_num/20)]
                num_noniid_labels = len(noniid_label_list)
                num_noniid_per_label = int(num_noniid/num_noniid_labels)
                num_noniid_per_label_last = num_noniid - (num_noniid_labels-1)*num_noniid_per_label

                label_count = 0
                for label in noniid_label_list:
                    label_count += 1
                    num_noniid_this_label = num_noniid_per_label
                    start = label * num_per_label_total + label_used[label]
                    if label_count == num_noniid_labels:
                        num_noniid_this_label = num_noniid_per_label_last
                    if (label_used[label] + num_noniid_this_label) > num_per_label_total:
                        start = label * num_per_label_total
                        label_used[label] = 0
                    dataidx_map[client] = np.concatenate((dataidx_map[client], idxs[start:start+num_noniid_this_label]), 
                                                    axis=0)
                    label_used[label] = label_used[label] + num_noniid_this_label
                dataidx_map[client] = dataidx_map[client].astype(int)
        
        else:
            raise NotImplementedError

        # assign data
        for client in range(num_clients):
            idxs = dataidx_map[client]
            X[client] = dataset_image[idxs]
            y[client] = dataset_label[idxs]

            for label in np.unique(y[client]):
                statistic[client].append((int(label), int(sum(y[client]==label))))
        del data

        for client in range(num_clients):
            print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
            print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            print("-" * 50)

        return X, y, statistic
    
    def _split_data(self, **kwargs):
        X = kwargs.get("images")
        y = kwargs.get("labels")
        # Split dataset
        train_data, test_data = [], []
        num_samples = {'train':[], 'test':[]}

        for i in range(len(y)):
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=self.train_ratio, shuffle=True)

            train_data.append({'x': X_train, 'y': y_train})
            num_samples['train'].append(len(y_train))
            test_data.append({'x': X_test, 'y': y_test})
            num_samples['test'].append(len(y_test))

        print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
        print("The number of train samples:", num_samples['train'])
        print("The number of test samples:", num_samples['test'])
        del X, y

        return train_data, test_data

    def _save_file(self,
              train_data, 
              test_data, 
              statistic):
        config = {
            'dataset': self.dataset, 
            'num_clients': self.num_clients, 
            'num_classes': self.num_classes, 
            'num_per_client': self.num_per_client,
            'partition': self.partition_method,
            'train_ratio': self.train_ratio,
            'Size of samples for labels in clients': statistic, 
        }

        print("\nSaving to disk...")

        for idx, train_dict in enumerate(train_data):
            with open(os.path.join(self.train_path, str(idx)+'.npz') , 'wb') as f:
                np.savez_compressed(f, data=train_dict)
        for idx, test_dict in enumerate(test_data):
            with open(os.path.join(self.test_path, str(idx)+'.npz'), 'wb') as f:
                np.savez_compressed(f, data=test_dict)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

        print("Finish generating dataset:\n")
        config.pop('Size of samples for labels in clients')
        print(config)
