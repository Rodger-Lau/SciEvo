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
    

# def get_dataloaders_scaler_and_split_task(dataset_dir, batch_size=16, task_per_dir=4, logger=None):
    
#     data = {}
#     scalers = []
#     datasets = {}
#     dataset = {}
#     dataloader = {}
#     num_samples = 0
#     dataloaders = []
#     train_tasks = {}
#     val_tasks = {}
#     test_tasks = {}
#     for i in range(task_per_dir):
#         train_tasks[i] = []
#         val_tasks[i] = []
#         test_tasks[i] = []
#     for category in ['train', 'val', 'test']:
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
#         data['x_' + category] = cat_data['x']
#         data['y_' + category] = cat_data['y'][...,:1]
#         num_samples += data['x_' + category].shape[0]
        
#     #scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
#     # Data format
#     # for category in ['train', 'val', 'test']:
#     #     data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
#         #datasets[category] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_' + category]), torch.FloatTensor(data['y_' + category]))
#     num_nodes = data['x_train'].shape[2]
#     # (num_samples, length, num_nodes, dim)
#     # logger.info(f"Data Length: {num_samples} Node num: {data['x_train'].shape[2]}")
#     # logger.info(f"Train num: {data['x_train'].shape[0]} Val num: {data['x_val'].shape[0]} Test num: {data['x_test'].shape[0]}")
#     for i in range(len(data['x_train'])):  # 根据timestamp划分
#         hour = data['x_train'][i,11,0,8]  # 取(i,11,0,8)作为判断小时的点
#         train_tasks[hour//(24//task_per_dir)].append(i)
#     for i in range(len(data['x_val'])):  # 根据timestamp划分
#         hour = data['x_val'][i,11,0,8]  # 取(i,11,0,8)作为判断小时的点
#         val_tasks[hour//(24//task_per_dir)].append(i)
#     for i in range(len(data['x_test'])):  # 根据timestamp划分
#         hour = data['x_test'][i,11,0,8]  # 取(i,11,0,8)作为判断小时的点
#         test_tasks[hour//(24//task_per_dir)].append(i)
#     for i in range(task_per_dir):
#         # dataset['train'] = datasets['train'][train_tasks[i]]
#         # dataset['val'] = datasets['val'][val_tasks[i]]
#         # dataset['test'] = datasets['test'][test_tasks[i]]
#         scaler = StandardScaler(mean=data['x_train'][train_tasks[i],..., 0].mean(), std=data['x_train'][train_tasks[i],..., 0].std())
#         data['x_train'][train_tasks[i], ..., 0] = scaler.transform(data['x_train'][train_tasks[i],..., 0])
#         data['x_val'][val_tasks[i], ..., 0] = scaler.transform(data['x_val'][val_tasks[i], ..., 0])
#         data['x_test'][test_tasks[i], ..., 0] = scaler.transform(data['x_test'][test_tasks[i], ..., 0])
#         dataset['train'] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_train'][train_tasks[i]]),
#                                                           torch.FloatTensor(data['y_train'][train_tasks[i]]))
#         dataset['val'] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_val'][val_tasks[i]]),
#                                                           torch.FloatTensor(data['y_val'][val_tasks[i]]))
#         dataset['test'] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_test'][test_tasks[i]]),
#                                                           torch.FloatTensor(data['y_test'][test_tasks[i]]))
#     #print(data['x_train'][0,:,0,1])
#         dataloader['train'] = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
#         dataloader['val'] = torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size, shuffle=False)
#         dataloader['test'] = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
#         dataloaders.append(dataloader)
#         scalers.append(scaler)
#     return dataloaders, scalers, num_nodes


# def get_scaler(data_loader, batch_size=16, task_per_dir=4, logger=None,mode=0):
#     if mode == 0:  # 若为loader形式，手动求均值方差
#         sum = 0.0
#         total_length = 0
#         length = []
#         mean = []
#         std = []
#         for data in data_loader:
#             input, real = data
#             mean.append(input[...,0].mean())
#             std.append(torch.pow(input[...,0].std(),2))
#             length.append(len(input))
#         for i in range(len(mean)):
#             sum += mean[i]*length[i]
#             total_length+=length[i]
#         data_mean = sum/total_length
#         sum = 0.0
#         for i in range(len(std)):
#             sum += length[i]*(std[i]+(mean[i]-data_mean)**2)
#         data_std = math.sqrt(sum/total_length)
#         scaler = StandardScaler(mean=data_mean,std=data_std)
#         return scaler
#     else:
#         scaler = StandardScaler(mean=data_loader[...,0].mean(),std=data_loader[...,0].std())
#         return scaler



def get_dataloaders_scaler_and_split_task(dataset_dir, batch_size=16, task_per_dir=4, logger=None):
    print("=== get_dataloaders_scaler_and_split_task START ===")
    data = {}
    # 1. 读取 train/val/test .npz 文件，并存到 data dict
    for category in ['train', 'val', 'test']:
        file_path = os.path.join(dataset_dir, category + '.npz')
        print(f"Loading {category} data from {file_path} ...")
        cat_data = np.load(file_path)
        
        data['x_' + category] = cat_data['x']              # shape: (num_samples, length, num_nodes, dim)
        data['y_' + category] = cat_data['y'][..., :1]     # 只取前 1 个输出特征

        print(f"{category} data shape: x_{category}={data['x_'+category].shape}, "
              f"y_{category}={data['y_'+category].shape}")

    # 2. 打印 num_nodes
    num_nodes = data['x_train'].shape[2]
    print(f"Number of nodes: {num_nodes}")

    # 3. 获取 hour 数组
    print("Extracting hours from x_train/x_val/x_test...")
    hour_train = data['x_train'][:, 11, 0, 8].astype(int)  # shape: (num_train_samples,)
    hour_val   = data['x_val'][:,   11, 0, 8].astype(int)  # shape: (num_val_samples,)
    hour_test  = data['x_test'][:,  11, 0, 8].astype(int)  # shape: (num_test_samples,)

    print(f"hour_train shape = {hour_train.shape}, hour_val shape = {hour_val.shape}, hour_test shape = {hour_test.shape}")

    # 4. 计算每个样本所属的分组 ID
    print("Calculating group IDs for each sample...")
    group_train = hour_train // (24 // task_per_dir)  # 每个样本属于哪个组
    group_val   = hour_val   // (24 // task_per_dir)
    group_test  = hour_test  // (24 // task_per_dir)

    # 5. 用 np.where() 找到各组的样本索引
    train_tasks = [np.where(group_train == i)[0] for i in range(task_per_dir)]
    val_tasks   = [np.where(group_val   == i)[0] for i in range(task_per_dir)]
    test_tasks  = [np.where(group_test  == i)[0] for i in range(task_per_dir)]

    for i in range(task_per_dir):
        print(f"Group {i} -> train_samples: {len(train_tasks[i])}, "
              f"val_samples: {len(val_tasks[i])}, test_samples: {len(test_tasks[i])}")

    dataloaders = []
    scalers = []

    # 6. 遍历每个分组并创建 DataLoader
    print("Creating DataLoaders for each group/task...")
    for i in range(task_per_dir):
        print(f"=== Now processing group {i} ===")
        sel_train_idx = train_tasks[i]
        sel_val_idx   = val_tasks[i]
        sel_test_idx  = test_tasks[i]

        # 如果该分组训练集是空的，做个提示
        if len(sel_train_idx) == 0:
            msg = f"[Warning] No training samples in group {i}."
            print(msg)
            if logger is not None:
                logger.warning(msg)
            # 如果要跳过这个分组:
            # continue
            # 或者继续往下也行，看你实际需求
            # 这里演示一下继续往下，但需要自己判断是否会影响后续流程

        # 6.1 创建 StandardScaler
        mean_val = data['x_train'][sel_train_idx, ..., 0].mean() if len(sel_train_idx) > 0 else 0
        std_val  = data['x_train'][sel_train_idx, ..., 0].std()  if len(sel_train_idx) > 0 else 1e-8
        scaler = StandardScaler(mean=mean_val, std=std_val)

        print(f"Scaler for group {i}: mean={scaler.mean}, std={scaler.std}")

        # 6.2 对 train/val/test 数据做 transform
        data['x_train'][sel_train_idx, ..., 0] = scaler.transform(data['x_train'][sel_train_idx, ..., 0])
        data['x_val'][sel_val_idx,     ..., 0] = scaler.transform(data['x_val'][sel_val_idx,     ..., 0])
        data['x_test'][sel_test_idx,   ..., 0] = scaler.transform(data['x_test'][sel_test_idx,   ..., 0])

        # 6.3 创建 Dataset
        print(f"Constructing TensorDataset for group {i}...")
        dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_train'][sel_train_idx]),
            torch.FloatTensor(data['y_train'][sel_train_idx])
        )
        dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_val'][sel_val_idx]),
            torch.FloatTensor(data['y_val'][sel_val_idx])
        )
        dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_test'][sel_test_idx]),
            torch.FloatTensor(data['y_test'][sel_test_idx])
        )

        # 6.4 创建 DataLoader
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val,   batch_size=batch_size, shuffle=False)
        dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

        dataloader_dict = {
            "train": dataloader_train,
            "val":   dataloader_val,
            "test":  dataloader_test
        }

        dataloaders.append(dataloader_dict)
        scalers.append(scaler)
        print(f"Group {i} DataLoader created: "
              f"train_size={len(dataset_train)}, val_size={len(dataset_val)}, test_size={len(dataset_test)}")

    print("=== get_dataloaders_scaler_and_split_task DONE ===")
    return dataloaders, scalers, num_nodes
