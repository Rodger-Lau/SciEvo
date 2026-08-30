import yaml
import sys
import os
import numpy as np
import scipy.sparse as sp

def create_random_adjacency(n, p=0.2):
    """创建随机邻接矩阵，p是节点间有边的概率"""
    adj = np.random.rand(n, n) < p
    adj = adj.astype(float)
    # 确保邻接矩阵对称（无向图）
    adj = (adj + adj.T) / 2
    # 清除对角线（没有自环）
    np.fill_diagonal(adj, 0)
    return adj

def normalize_adj(adj):
    """规范化邻接矩阵，添加自环并归一化"""
    adj = adj + np.eye(adj.shape[0])  # 添加自环
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


proj_dir = os.path.dirname(os.path.abspath(__file__))
print(proj_dir)
sys.path.append(proj_dir)
conf_fp = os.path.join(proj_dir, 'config.yaml')
with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


nodename = os.uname().nodename
print(nodename)
#file_dir = config['filepath'][nodename]
file_dir = config['filepath']['GPU-Server']

def main():
    pass


if __name__ == '__main__':
    main()
