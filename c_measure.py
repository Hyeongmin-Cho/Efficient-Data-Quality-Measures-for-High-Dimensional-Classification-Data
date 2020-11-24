import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd
import sys
import smtplib

import math
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import dataset
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset
from torchvision import datasets, transforms
import pickle
from tqdm.notebook import tqdm
from datetime import datetime
from utils import load_list, save_list, to_arff
import PIL

class StratifiedSampler(Sampler):
    """Stratified Sampling

    Provides equal representation of target classes in each batch
    """
    
    # self.label_dict : { class : [idx] }
    # self.class_instance_dict : { class : # of data (if class data less than 30, then 0) }
    # self.batch_size : batch_sampling_number
    # self.min_class_data_size : minimum number of data in a class
    def __init__(self, label_dict, sampling_count, class_instance_dict, batch_size, min_class_data_size):
        self.label_dict = label_dict
        self.class_instance_dict = class_instance_dict
        self.batch_size = batch_size
        self.sampling_count = sampling_count
        self.min_class_data_size = min_class_data_size
        
    def gen_sample_idx(self):
        sample_idx = []
        
        # Iteration (sampling count)
        for _ in range(self.sampling_count):
            tmp_idx = []
            
            for label in self.label_dict.keys():
                if self.class_instance_dict[label] == 0:
                    continue
                else:
                    stratified_idx = np.random.choice(self.label_dict[label], self.class_instance_dict[label], replace=True)
                    tmp_idx.extend(stratified_idx)
                
            sample_idx.append(tmp_idx)
        
        print('Sampler index size :', len(tmp_idx))
        return np.array(sample_idx).reshape(-1)
        
        
    def __iter__(self):
        return iter(self.gen_sample_idx())

    def __len__(self):
        return len(self.class_vector)

class DatasetQualityEval():
    # MNIST : 1000 x 784 ( # of data x flatten features)
    # Input : numpy array
    def __init__(self, loader, process= 1 ,resize = 1, sample_ratio = 0.3, sampling_count = 2, normal_vector = 10, batch_size = 100, dataset_name = 'NoName', size = (256, 256, 3)):

        # data 
        self.loader = loader
        
        self.size = size
        self.resize = resize
        self.process = process
        
        # sampling ratio (train data = total # of data * sampling ratio)
        # sampling count : # of bootstrapping
        self.sample_ratio =  sample_ratio
        self.sampling_count = sampling_count
        
        self.dset_name = dataset_name
        
        # flatten data dimension
        self.data_dim = size[0] * size[1] * size[2]
        
        # coherence (LDA)
        self.coherence_result = 0
        self.avg_Sb = 0
        self.avg_Sw = 0
        
        self.normal_vec_num = normal_vector
        self.between_vect_mean = {}
        
        self.LDA_list = []
        self.max_LDA_list = []
        self.SB_list = []
        self.SW_list = []
        self.S_B_variance_list = []
        self.S_W_variance_list = []
        self.batch_size = batch_size
        self.random_vector_list = []
        
        
    def cal_lda(self, sampled_X_data, sampled_Y_data):
        sampled_X_data = sampled_X_data.astype('float32')
        sampled_Y_data = sampled_Y_data.astype('int')
        mean = sampled_X_data.mean()
        std = sampled_X_data.std()
        sampled_X_data = (sampled_X_data - mean) / std

        add = {}
        count = {}
        S_W_variance = {}
        S_B_variance = {}
        S_B_list = []
        S_W_list = []
        LDA_list = []
        # calculate # of class data (count) and class sum (add)
        for idx in range(sampled_Y_data.shape[0]):
            label = sampled_Y_data[idx]
            count[label] = 1 if label not in list(count.keys()) else count[label] + 1
            add[label] = sampled_X_data[idx] if label not in list(add.keys()) else add[label] + sampled_X_data[idx]

        # calculate each class mean (X bar i) and global mean (X bar)
        each_class_mean = {each_class : (add[each_class]/count[each_class]).reshape(self.data_dim, 1) for each_class in list(add.keys())}
        global_mean = np.mean(sampled_X_data, axis=0).reshape(self.data_dim, 1)
        
        gaussian_vec = np.random.normal(0, 1, (self.normal_vec_num, self.data_dim))
        max_lda, max_s_b, max_s_w, max_gaussian = 0, 0, 0, 0
        
        # Matrix computation is inefficient
        # (# class x data_dim) -->  matrix memory issue
        # compute each class, not at once
        for one_gaussian_vec in gaussian_vec:
            S_B = 0
            S_W = 0
            # projection vector is unit vector
            one_gaussian_vec = one_gaussian_vec / np.linalg.norm(one_gaussian_vec)
            
            self.random_vector_list.append(one_gaussian_vec)
            
            for label, mean_vec in each_class_mean.items():
                n = count[label]
                
                # between class with normalization
                between_class = (n / sampled_Y_data.shape[0]) * np.matmul(one_gaussian_vec.T, (mean_vec - global_mean)) * np.matmul((mean_vec - global_mean).T , one_gaussian_vec)
                S_B_variance[label] = between_class / self.normal_vec_num if label not in S_B_variance.keys() else S_B_variance[label] + (between_class / self.normal_vec_num)

                # within class with normalization
                label_instance = sampled_X_data[np.where(sampled_Y_data == label)]
                within_class = np.matmul( np.matmul(one_gaussian_vec.T , (label_instance - mean_vec.T).T), np.matmul( (label_instance - mean_vec.T), one_gaussian_vec)) / n
                
                # Save SW values (v S_w_hat v^T) for the calculation of M_var
                if label not in S_W_variance.keys():
                    S_W_variance[label] = [within_class / self.normal_vec_num]
                else:
                    S_W_variance[label].append(within_class / self.normal_vec_num)
                
                S_W += abs(within_class)
                S_B += abs(between_class)
            
            # save --> S_w, S_b, lda
            S_W /= len(each_class_mean.items())
            S_B /= len(each_class_mean.items())
            
            lda = S_B / S_W
            S_W_list.append(S_W.item())
            S_B_list.append(S_B.item())
            LDA_list.append(lda.item())
            
            dtime = datetime.fromtimestamp(time.time())
            # save log files
            f_sb = open('./log/%s_resize%d_ratio%f_count%d_gvn%d_SB_log.txt'%(self.dset_name, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='a+', encoding='utf-8')
            f_sw = open('./log/%s_resize%d_ratio%f_count%d_gvn%d_SW_log.txt'%(self.dset_name, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='a+', encoding='utf-8')
            f_lda = open('./log/%s_resize%d_ratio%f_count%d_gvn%d_lda_log.txt'%(self.dset_name, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='a+', encoding='utf-8')
            f_sb.write('%s_%d\t%s\n' % (dtime, self.process, S_B.item())) 
            f_sw.write('%s_%d\t%s\n' % (dtime, self.process, S_W.item()))
            f_lda.write('%s_%d\t%s\n' % (dtime, self.process, lda.item()))
            f_sb.close()
            f_sw.close()
            f_lda.close()
            
            # update max lda
            if lda > max_lda:
                max_lda = lda
                max_s_b = S_B
                max_s_w = S_W
                max_gaussian = one_gaussian_vec

        
        f_sb_variance = open('./log/%s_resize%d_ratio%f_count%d_gvn%d_SB_variance.txt'%(self.dset_name, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='a+', encoding='utf-8')
        f_sb_variance.write('%s_%d\t%s\n' % (dtime, self.process, S_B_variance))
        f_sb_variance.close()
        
        
        return S_B_variance, S_W_variance, S_W_list, S_B_list, LDA_list
    
    
    def coherence(self, normal_vec_num = 100):
        start_time = time.time()
        self.normal_vec_num = normal_vec_num

        for data, label in tqdm(self.loader):
            iter_start_time = time.time()
            data = np.array(data).reshape(self.batch_size, -1)
            label = np.array(label)
            S_B_variance, S_W_variance, SW, SB, LDA = self.cal_lda(data, label)
            self.S_B_variance_list.append(S_B_variance)
            self.S_W_variance_list.append(S_W_variance)
            self.SW_list.extend(SW)
            self.SB_list.extend(SB)
            self.LDA_list.extend(LDA)
            self.max_LDA_list.append(np.max(LDA))
            iter_end_time = time.time()
        
        # Bootstrapping statistic
        self.coherence_result = np.mean(self.max_LDA_list)
        
        
        end_time = time.time()
        elapsed_time = end_time-start_time
        
        # Save summarized results
        print("\nSample ratio: %.2f, Sampling count: %d" %(self.sample_ratio, self.sampling_count))
        print("Data coherence:", self.coherence_result)
        print("Computing time: %d hour %d min %d sec (%.3f)\n" % (elapsed_time/3600, (elapsed_time%3600)/60, elapsed_time%60, elapsed_time))
        f = open('./log/%s_process%d_resize%d_ratio%f_count%d_gvn%d.txt'%(self.dset_name, self.process, self.resize, self.sample_ratio, self.sampling_count, self.normal_vec_num), mode='wt', encoding='utf-8')
        f.write('resize : 1/%d\n' % (self.resize))
        f.write('Sampled data : %d\n' % (self.batch_size) )
        f.write("Sample ratio: %.2f, Sampling count: %d\n" %(self.sample_ratio, self.sampling_count))
        f.write("Num Normal vector : %d \n" % (self.normal_vec_num))
        f.write("Data coherence : %.8f\n" % self.coherence_result)
        
        f.write("Computing time: %d hour %d min %d sec (%.3f)\n" % (elapsed_time/3600, (elapsed_time%3600)/60, elapsed_time%60, elapsed_time))
        f.close()
        
        return self.coherence_result
    
    # deprecated function
    def between_class_mean(self):
        class_labels = list(set(j  for i in self.S_B_variance_list for j in i.keys()))
        class_count = {}
        add_vec = {}
        for label in class_labels:
            for i in range(len(self.S_B_variance_list)):
                if label in (self.S_B_variance_list[i].keys()):
                    class_count[label] = 1 if label not in class_count.keys() else class_count[label] + 1
                    add_vec[label] = self.S_B_variance_list[i][label] if label not in add_vec.keys() else add_vec[label] + self.S_B_variance_list[i][label]
        
        self.between_vect_mean = {label : (add / float(class_count[label])) for label, add in add_vec.items()}
        return self.between_vect_mean    
    

# Data loader
def load_data(root, dataset_name, is_training):
    dataset_path = os.path.join(root, dataset_name)

    load = dataset.dataloader()
    if dataset_name == 'mnist':
        data, label = load.load_mnist(dataset_path, training=is_training)
        data = data / 255
        size = (28, 28, 1)
    elif dataset_name == 'cifar10':
        data, label = load.load_cifar10(dataset_path, training=is_training)
        data = data / 255
        size = (32, 32, 3)
    elif dataset_name == 'notMNIST':
        data, label = load.load_notMNIST(dataset_path, training=is_training)
        size = (28, 28, 3)
    elif dataset_name == 'stl10':
        data = []
        label = []
        train_loader = load.load_stl10(dataset_path, training=True)
        test_loader = load.load_stl10(dataset_path, training=False)

        for i in tqdm(range(len(train_loader))):
            data.append(np.array(train_loader[i][0]))
            label.append(train_loader[i][1])

        for i in tqdm(range(len(test_loader))):
            data.append(np.array(test_loader[i][0]))
            label.append(test_loader[i][1])

        size = (32, 32, 3)
        data = np.array(data)
        label = np.array(label)
        data = data / 255
        print(data.shape, label.shape)
    else:
        data, label = load.load_linnaeus(dataset_path, training=is_training)
        size = (32, 32, 3)

    num_data = data.shape[0]
    data = data.reshape(num_data, -1)
    print('Before sampling :', data.shape, label.shape)
    data, label = data[:10000], label[:10000]
    print('After sampling :', data.shape, label.shape)
    
    return data, label, size


def get_lda_object(root, dataset_name, is_training, sample_ratio=1, sampling_count=100, normal_vector=10, min_sampling_num=30 ,num_workers=1, process=1, resize=1):
    # fix seed
    np.random.seed(2) 
    data, label, size = load_data(root, dataset_name, is_training)
    print('Loading dataset', dataset_name, '...')
    print('Data min :', data.min())
    print('Data max :', data.max())
    print('Data mean :', data.mean())
    print('Data std :', data.std())
    
    print('\nSampling ratio :', sample_ratio)
    print('Sampling count :', sampling_count)
    print('Num normal vector :', normal_vector)
    
    # calcuate batch size
    standard_batch_size = int(data.shape[0] * sample_ratio)


    print('Standard batch size :', standard_batch_size)

    # calculate # of class
    unique_label = list(set(label))

    label_count_dict = {key : 0 for key in unique_label}
    for l in label:
        label_count_dict[l] += 1

    label_idx_dict = {key : [] for key in unique_label}
    for idx, l in enumerate(label):
        label_idx_dict[l].append(idx)
        
    # standard_batch_size : batch size
    # label_idx_dict : index dictionary of class label
    # label_count_dict : # of data in each class
    # label_stratified_sampling_num : Dictionary of # of class data into the batch
    # batch_sampling_num : Toal # of data to be sampled
    # min_sampling_num : minimum sampling #, No statistical significance below 30

    label_stratified_sampling_num = {}
    batch_sampling_num = 0
    total_data_num = len(label)

    for key, val in label_count_dict.items():
        label_stratified_sampling_num[key] = round(val / total_data_num * standard_batch_size)

        # Sampling with replacement when the number of data in the class is less than the minimum number of samples
        if label_stratified_sampling_num[key] < min_sampling_num :
            label_stratified_sampling_num[key] = min_sampling_num

        batch_sampling_num += label_stratified_sampling_num[key]

    print('Batch sampling num :', batch_sampling_num)

    # Define StratifiedSampler
    sampler = StratifiedSampler(label_dict=label_idx_dict, sampling_count = sampling_count, class_instance_dict = label_stratified_sampling_num,
                                    batch_size = batch_sampling_num, min_class_data_size = min_sampling_num)

    dset = TensorDataset(torch.Tensor(data), torch.Tensor(label))
    del data, label

    # Define dataloader from pytorch DataLoader
    loader = DataLoader(dset, batch_size=batch_sampling_num, shuffle=False, num_workers=num_workers, sampler=sampler)
    
    # Define python object for Dataquality measure
    indicator = DatasetQualityEval(loader, process=process, resize=resize, sample_ratio=sample_ratio, \
                               sampling_count=sampling_count, normal_vector=normal_vector, \
                               batch_size=loader.batch_size, dataset_name=dataset_name, \
                              size = size)
    return indicator