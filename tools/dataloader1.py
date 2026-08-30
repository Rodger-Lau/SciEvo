import math

import torch
import numpy as np
import os

class StandardScaler:
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    

def get_dataloaders_scaler_and_split_task(dataset_dir, batch_size=16, task_per_dir=4, logger=None):
    
    data = {}
    scalers = []
    datasets = {}
    dataset = {}
    dataloader = {}
    num_samples = 0
    dataloaders = []
    train_tasks = {}
    val_tasks = {}
    test_tasks = {}
    for i in range(task_per_dir):
        train_tasks[i] = []
        val_tasks[i] = []
        test_tasks[i] = []
    for category in ['train', 'val', 'test']:
        print(dataset_dir,category)
        print(os.path.join(dataset_dir, category + '.npz'))
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y'][...,:1]
        num_samples += data['x_' + category].shape[0]
        
    #scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
    # Data format
    # for category in ['train', 'val', 'test']:
    #     data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        #datasets[category] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_' + category]), torch.FloatTensor(data['y_' + category]))
    num_nodes = data['x_train'].shape[2]
    # (num_samples, length, num_nodes, dim)
    # logger.info(f"Data Length: {num_samples} Node num: {data['x_train'].shape[2]}")
    # logger.info(f"Train num: {data['x_train'].shape[0]} Val num: {data['x_val'].shape[0]} Test num: {data['x_test'].shape[0]}")
    for i in range(len(data['x_train'])):  # 根据timestamp划分
        hour = data['x_train'][i,11,0,8]  # 取(i,11,0,8)作为判断小时的点
        train_tasks[hour//6].append(i)
        # month = data['x_train'][i,11,0,6]
        # train_tasks[month-1].append(i)
    for i in range(len(data['x_val'])):  # 根据timestamp划分
        hour = data['x_val'][i,11,0,8]  # 取(i,11,0,8)作为判断小时的点
        val_tasks[hour//6].append(i)
        # month = data['x_val'][i,11,0,6]
        # val_tasks[month-1].append(i)
    for i in range(len(data['x_test'])):  # 根据timestamp划分
        hour = data['x_test'][i,11,0,8]  # 取(i,11,0,8)作为判断小时的点
        test_tasks[hour//6].append(i)
        # month = data['x_test'][i,11,0,6]
        # test_tasks[month-1].append(i)
    for i in range(task_per_dir):
        # dataset['train'] = datasets['train'][train_tasks[i]]
        # dataset['val'] = datasets['val'][val_tasks[i]]
        # dataset['test'] = datasets['test'][test_tasks[i]]
        scaler = StandardScaler(mean=data['x_train'][train_tasks[i],..., 0].mean(), std=data['x_train'][train_tasks[i],..., 0].std())
        data['x_train'][train_tasks[i], ..., 0] = scaler.transform(data['x_train'][train_tasks[i],..., 0])
        data['x_val'][val_tasks[i], ..., 0] = scaler.transform(data['x_val'][val_tasks[i], ..., 0])
        data['x_test'][test_tasks[i], ..., 0] = scaler.transform(data['x_test'][test_tasks[i], ..., 0])
        dataset['train'] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_train'][train_tasks[i]]),
                                                          torch.FloatTensor(data['y_train'][train_tasks[i]]))
        dataset['val'] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_val'][val_tasks[i]]),
                                                          torch.FloatTensor(data['y_val'][val_tasks[i]]))
        dataset['test'] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_test'][test_tasks[i]]),
                                                          torch.FloatTensor(data['y_test'][test_tasks[i]]))
    #print(data['x_train'][0,:,0,1])
        dataloader['train'] = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        dataloader['val'] = torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size, shuffle=False)
        dataloader['test'] = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
        dataloaders.append(dataloader)
        scalers.append(scaler)
    return dataloaders, scalers, num_nodes


def get_scaler(data_loader, batch_size=16, task_per_dir=4, logger=None,mode=0):
    if mode == 0:  # 若为loader形式，手动求均值方差
        sum = 0.0
        total_length = 0
        length = []
        mean = []
        std = []
        for data in data_loader:
            input, real = data
            mean.append(input[...,0].mean())
            std.append(torch.pow(input[...,0].std(),2))
            length.append(len(input))
        for i in range(len(mean)):
            sum += mean[i]*length[i]
            total_length+=length[i]
        data_mean = sum/total_length
        sum = 0.0
        for i in range(len(std)):
            sum += length[i]*(std[i]+(mean[i]-data_mean)**2)
        data_std = math.sqrt(sum/total_length)
        scaler = StandardScaler(mean=data_mean,std=data_std)
        return scaler
    else:
        scaler = StandardScaler(mean=data_loader[...,0].mean(),std=data_loader[...,0].std())
        return scaler


