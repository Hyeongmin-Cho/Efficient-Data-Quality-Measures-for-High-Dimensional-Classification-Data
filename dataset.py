import os
import numpy as np
import pickle
from torchvision import datasets, transforms
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from utils import load_list, save_list, unpickle
from torchvision.datasets import ImageFolder
from torchvision.datasets import SVHN

class dataloader:
    def __init__(self):
        pass
    
    def mnist_gen(self, path, training):
        if training:
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        elif not training :
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
        else:
            raise Exception("Training argument must be True or False")

        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        get_img = lambda idx: (lbl[idx], img[idx])

        # Create an iterator which returns each image in turn
        for i in range(len(lbl)):
            yield get_img(i)
    
    # MNIST, FMNIST Dataset
    def load_mnist(self, path ,training=True):
        print(path)
        data = list(self.mnist_gen(path, training))
        flatted_pixel_list = list()
        label_list = list()

        for i in trange(len(data)):
            label, pixel = data[i]
            flatted_pixel_list.append(np.array(pixel).flatten())
            label_list.append(label)

        return np.array(flatted_pixel_list), np.array(label_list)
    
    # SVHN Dataset
    def load_svhn(self, dataset_path, training=True):
        transform = transforms.Compose([
            transforms.ToTensor(), 
        ])
        if training:
            train_dataset = SVHN(dataset_path, split='train' ,transform=transform, download=True)
        else:
            train_dataset = SVHN(dataset_path, split='test' ,transform=transform, download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=False, drop_last=True)
        data = []
        target = []

        for X, y in train_loader:
            data.append(X.numpy())
            target.append(y.numpy())

        return np.array(data).reshape(-1, 3, 32, 32), np.array(target).reshape(-1)
    
    # Caltech256 dataset
    def load_caltech256(self):
        data = load_list('./dataset/caltech256_data_resized32.pkl')
        target = load_list('./dataset/caltech256_target.pkl')

        return data, target
    
    # Tiny ImageNet Dataset
    def load_imagenet32(self, types=1):
        data = []
        target = []

        print('Imagenet type%d 10 class dataset loading..'%types)
        data = np.array(load_list('./dataset/tiny_imagenet_type%d_data.pkl'%types))
        target = np.array(load_list('./dataset/tiny_imagenet_type%d_label.pkl'%types))

        return data, target
    
    # notMNIST Dataset
    def load_notMNIST(self, path, training=True):
        print(path)
        notMNIST = datasets.ImageFolder(root=path, transform=transforms.ToTensor())
        dataset_loader = torch.utils.data.DataLoader(notMNIST, batch_size=len(notMNIST.targets), shuffle=False)
        data, label = iter(dataset_loader).next()
        data, label = data.numpy(), label.numpy()
        
        idx = np.arange(data.shape[0])
        num_train = int(idx.shape[0] * 0.7)
        np.random.shuffle(idx)
        data, label = data[idx], label[idx]
        
        if training:
            return data[:num_train], label[:num_train]
        else:
            return data[num_train:], label[num_train:]
        
    # Linnaeus Dataset 
    def load_linnaeus(self, path, training=True):
        path = os.path.join(path, 'train') if training else os.path.join(path, 'test')
        print(path)
        linnaeus = datasets.ImageFolder(root=path, transform=transforms.ToTensor())
        dataset_loader = torch.utils.data.DataLoader(linnaeus, batch_size=len(linnaeus.targets), shuffle=False)
        data, label = iter(dataset_loader).next()
        data, label = data.numpy(), label.numpy()
        
        idx = np.arange(data.shape[0])
        num_train = int(idx.shape[0] * 0.7)
        np.random.shuffle(idx)
        data, label = data[idx], label[idx]
        
        if training:
            return data[:num_train], label[:num_train]
        else:
            return data[num_train:], label[num_train:]
        
        
    # Cifar10 Dataset
    def load_cifar10(self, path, training):
        print(path)
        data, label = np.array([]), np.array([])

        if training:
            path = os.path.join(path, 'data_batch_%d')
            for i in range(1, 6):
                tmp_data, tmp_label = unpickle(path%i)[b'data'], unpickle(path%i)[b'labels']
                tmp_data, tmp_label = np.array(tmp_data), np.array(tmp_label)
                data = np.vstack([data, tmp_data]) if data.size else tmp_data
                label = np.hstack([label, tmp_label]) if label.size else tmp_label
        else:
            path = os.path.join(path, 'test_batch')
            data, label = unpickle(path)[b'data'], unpickle(path)[b'labels']
            data, label = np.array(data), np.array(label)

        return data, label
    
    # STL10 Dataset
    def load_stl10(self, path, training):
        print(path)
        if training:
            stl10_dataset = datasets.STL10(path, split='train',transform=transforms.Resize(32))
        else:
            stl10_dataset = datasets.STL10(path, split='test',transform=transforms.Resize(32))
        
        return stl10_dataset