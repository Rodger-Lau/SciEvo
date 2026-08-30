import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
#import hdbscan

def clustering_dict(n_clusters, data_dict, algorithm='auto'):
    """
    对字典中的tensor值进行聚类，返回与原始键对应的聚类结果
    
    参数:
    n_clusters (int): 目标聚类数量
    data_dict (dict): 字典，格式为 {key: tensor}
    algorithm (str): 可选算法 ['auto', 'kmeans', 'minibatch', 'dbscan', 'hdbscan']
    
    返回:
    tuple: (cluster_labels, cluster_members)
    cluster_labels (dict): 键为原始字典键，值为聚类标签
    cluster_members (dict): 键为聚类标签，值为该簇中原始字典键的列表
    """
    # 1. 提取键和值
    keys = list(data_dict.keys())
    tensors = [data_dict[k] for k in keys]
    
    # 2. 将Tensor列表转换为NumPy数组
    data = []
    for tensor in tensors:
        # 将Tensor转换为NumPy数组并展平为一维向量
        if tensor.is_cuda:
            arr = tensor.cpu().detach().numpy().flatten()
        else:
            arr = tensor.detach().numpy().flatten()
        data.append(arr)
    
    # 3. 转换为NumPy矩阵
    data_matrix = np.vstack(data)
    n_samples, n_features = data_matrix.shape
    
    # 4. 算法选择逻辑
    if algorithm == 'auto':
        if n_samples > 10000:  # 大数据集
            algorithm = 'minibatch'
        elif n_features > 100:  # 高维数据
            algorithm = 'hdbscan'
        else:
            algorithm = 'kmeans'
    
    # 5. 执行聚类
    if algorithm == 'kmeans':
        #kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data_matrix)
        labels = kmeans.labels_
    
    elif algorithm == 'minibatch':
        mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, 
                             n_init='auto', batch_size=1024)
        mbk.fit(data_matrix)
        labels = mbk.labels_
    
    elif algorithm == 'dbscan':
        db = DBSCAN(eps=0.5, min_samples=5)
        db.fit(data_matrix)
        labels = db.labels_
        unique_labels = set(labels)
        n_found_clusters = len(unique_labels) - (-1 in unique_labels)
        
        if n_found_clusters < n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
            kmeans.fit(data_matrix)
            labels = kmeans.labels_
    
    # elif algorithm == 'hdbscan':
    #     clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    #     clusterer.fit(data_matrix)
    #     labels = clusterer.labels_
    #     unique_labels = set(labels)
    #     n_found_clusters = len(unique_labels) - (-1 in unique_labels)
        
    #     if n_found_clusters < n_clusters:
    #         kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    #         kmeans.fit(data_matrix)
    #         labels = kmeans.labels_
    
    else:
        raise ValueError(f"未知算法: {algorithm}")
    
    # 6. 构建结果字典
    cluster_labels = {}  # 键为原始键，值为聚类标签
    cluster_members = {}  # 键为聚类标签，值为该簇中的原始键列表
    
    # 为每个原始键分配聚类标签
    for i, key in enumerate(keys):
        cluster_labels[key] = int(labels[i])
        
        # 将原始键添加到对应簇的成员列表中
        label = int(labels[i])
        if label not in cluster_members:
            cluster_members[label] = []
        cluster_members[label].append(key)
    
    return cluster_labels, cluster_members

def get_cluster_average(cluster_members, cat_grad_dict):
    cluster_grad={}
    for cluster, members in cluster_members.items():
        print(cluster, members)
        # cluster_grad[cluster]=0
        # for member in members:
        #     cluster_grad[cluster] += cat_grad_dict[member]
        # cluster_grad[cluster] /= len(members)
        cluster_grad[cluster] = sum(cat_grad_dict[member] for member in members) / len(members)
    return cluster_grad