import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
class GCN(torch.nn.Module):
    def __init__(self, 
                 in_channels=7,     # 输入特征维度
                 hidden_dim=128,    # 隐藏层维度
                 output_dim=1,    # 输出维度（分类数）
                 drop_out=0.5):  # Dropout率
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)   # 第一层图卷积
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # 第二层图卷积
        self.dropout = drop_out
        self.classifier1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),)
        self.classifier2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)  # 二分类
        )
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层卷积 + ReLU激活 + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier1(x)
        x = F.relu(x)
        x = self.classifier2(x)
        # return F.log_softmax(x, dim=1)  # 输出log概率
        return x.squeeze(),0