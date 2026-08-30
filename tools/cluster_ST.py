import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
def clustering_2d_list(n_clusters, tensor_2d_list, algorithm='auto', eps=0.5, min_samples=5):
    """
    对二维列表中的tensor进行聚类，返回每个簇对应的位置坐标
    
    参数:
    n_clusters (int): 目标聚类数量
    tensor_2d_list (list of lists): 二维列表，每个元素是一个tensor
    algorithm (str): 可选算法 ['auto', 'kmeans', 'minibatch']
    
    返回:
    dict: 键为聚类标签，值为该簇中所有位置坐标的列表
           例如: {0: [(0,1), (0,0)], 1: [(1,0), (1,1)]}
    """
    # 1. 检查输入是否为二维列表
    if not isinstance(tensor_2d_list, list) or not all(isinstance(row, list) for row in tensor_2d_list):
        raise ValueError("输入必须是二维列表")
    
    # 2. 收集所有tensor及其位置
    positions = []  # 存储位置坐标 (row_idx, col_idx)
    tensors = []    # 存储对应的tensor

    for row_idx, row in enumerate(tensor_2d_list):
        for col_idx, tensor in enumerate(row):
            positions.append((row_idx, col_idx))
            tensors.append(tensor)
    
    # 3. 如果没有tensor，返回空字典
    if len(tensors) == 0:
        return {}
    
    # 4. 将tensor列表转换为NumPy数组
    data = []
    for tensor in tensors:
        if tensor.is_cuda:
            arr = tensor.cpu().detach().numpy().flatten()
        else:
            arr = tensor.detach().numpy().flatten()
        data.append(arr)
    
    data_matrix = np.vstack(data)

    scaler = StandardScaler()
    data_matrix_scaled = scaler.fit_transform(data_matrix)
    # 5. 算法选择
    if algorithm == 'auto':
        if len(tensors) > 10000:
            algorithm = 'minibatch'
        else:
            algorithm = 'kmeans'
    
    # 6. 执行聚类
    if algorithm == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        kmeans.fit(data_matrix)
        labels = kmeans.labels_
    
    elif algorithm == 'minibatch':
        mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, 
                             n_init='auto', batch_size=1024)
        mbk.fit(data_matrix)
        labels = mbk.labels_
    elif algorithm == 'dbscan':
        # DBSCAN不需要指定簇数
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data_matrix_scaled)
        labels = dbscan.labels_
    
    else:
        raise ValueError(f"不支持算法: {algorithm}")
    
    # 7. 构建结果字典：簇标签 -> 位置列表
    cluster_positions = {}
    
    for pos, label in zip(positions, labels):
        #po = [pos[0],pos[1]]
        label = int(label)  # 转换为Python整数
        if label not in cluster_positions:
            cluster_positions[label] = []
        cluster_positions[label].append(pos)
    
    return cluster_positions

def get_cluster_average(cluster_members, cat_grad_dict):
    cluster_grad={}
    for cluster, members in cluster_members.items():
        print(cluster, members)
        # cluster_grad[cluster]=0
        # for member in members:
        #     cluster_grad[cluster] += cat_grad_dict[member]
        # cluster_grad[cluster] /= len(members)
        cluster_grad[cluster] = sum(cat_grad_dict[member[0]][member[1]] for member in members) / len(members)
    return cluster_grad