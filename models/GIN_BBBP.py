import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Module, Dropout
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import os
import shutil
from torch_geometric.nn import GATConv
class GAT(nn.Module):
    def __init__(self,
                 in_channels,       # 输入特征维度
                 hidden_channels,   # 隐藏层维度
                 output_dim=2,      # 输出维度
                 num_heads=8,       # 多头注意力头数
                 dropout=0.6,       # Dropout率
                 negative_slope=0.2, # LeakyReLU负斜率
                 residual=True):    # 是否使用残差连接
        super().__init__()
        out_channels = output_dim
        # 第一层：多头图注意力层
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=num_heads, 
            dropout=dropout,
            negative_slope=negative_slope,
            concat=True
        )
        
        # 第二层：多头图注意力层（输出层）
        self.conv2 = GATConv(
            hidden_channels * num_heads,  # 第一层输出维度 = 隐藏层维度 * 头数
            hidden_channels, 
            heads=1,  # 输出层通常只使用一个头
            dropout=dropout,
            negative_slope=negative_slope,
            concat=False  # 输出层不拼接多头结果
        )
        
        self.drop_out = dropout
        self.negative_slope = negative_slope
        self.residual = residual
        
        # 残差连接的线性变换
        if residual:
            self.residual_linear = nn.Linear(in_channels, hidden_channels * num_heads)
        else:
            self.register_parameter('residual_linear', None)
        self.classifier1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),)
        self.classifier2 = nn.Sequential(
            nn.Linear(hidden_channels, out_channels)  # 二分类
        )
    def forward(self, data,activate_num=0):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_initial = x  # 保存初始特征用于残差连接
        count = 0
        self.Dropout = nn.Dropout(self.drop_out)
        # 第一层GAT
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # 使用ELU激活函数
        
        # 残差连接
        if self.residual and self.residual_linear is not None:
            res = self.residual_linear(x_initial)
            x = x + res
        
        x = self.Dropout(x)
        #x = global_add_pool(x, batch)
        # 第二层GAT
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier1(x)
        size1 = x.shape[0] * x.shape[1] 
        count += (x > 0).sum().item()
        activate_num += count / size1
        x = F.relu(x)
        x = self.classifier2(x)
        #x= F.log_softmax(x, dim=1)
        # print(x.shape)
        # return F.log_softmax(x, dim=1)  # 输出log概率
        return  x.squeeze() , activate_num

# 可选：更深的GAT模型（三层）
class DeepGAT(GAT):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_heads=8, 
                 dropout=0.6,
                 negative_slope=0.2,
                 residual=True):
        super().__init__(in_channels, hidden_channels, out_channels, num_heads, dropout, negative_slope, residual)
        
        # 增加中间层
        self.mid_conv = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            negative_slope=negative_slope,
            concat=True
        )
        
        # 中间层残差连接
        if residual:
            self.mid_residual = nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        else:
            self.register_parameter('mid_residual', None)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_initial = x
        
        # 第一层
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # 第一层残差
        if self.residual and self.residual_linear is not None:
            res = self.residual_linear(x_initial)
            x = x + res
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 中间层
        x_mid = x
        x = self.mid_conv(x, edge_index)
        x = F.elu(x)
        
        # 中间层残差
        if self.residual and self.mid_residual is not None:
            res = self.mid_residual(x_mid)
            x = x + res
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 输出层
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.5):
        super().__init__()
        #self.input = Linear(in_channels, hidden_dim)
        self.num_layers = num_layers
        self.Relu = nn.LeakyReLU()
        self.drop_out = dropout
        self.Dropout = nn.Dropout(dropout)
        # 使用MLP作为GIN的聚合函数
        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        #self.convs = nn.ModuleList()
        # for _ in range(num_layers):
        #     mlp = nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.BatchNorm1d(hidden_dim),
        #         nn.ReLU(),
        #         nn.Dropout(self.drop_out),
        #         nn.Linear(hidden_dim, hidden_dim)
        #     )
        #     self.convs.append(GINConv(mlp, train_eps=True))
        for _ in range(num_layers):
            mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU())
            self.convs1.append(GINConv(mlp, train_eps=True))
            #self.convs1.append(mlp)
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim)
            )
            self.convs2.append(GINConv(mlp, train_eps=True))
            #self.convs2.append(mlp)
        # 分类器
        self.classifier1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),)
        self.classifier2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)  # 二分类
        )
        
        # 初始节点特征转换
        self.initial_lin = nn.Linear(in_channels, hidden_dim)
    
    def forward(self, data, activate_num=0):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        count = 0
        self.Dropout = nn.Dropout(self.drop_out)
        #self.Relu = nn.ReLU()
        x = self.initial_lin(x)
        #self.convs = nn.ModuleList()
        # for i in range(self.num_layers):
        #     mlp = nn.Sequential(
        #         self.convs1[i],
        #         self.Dropout,
        #         self.convs2[i]
        #     )
        #     self.convs.append(mlp)
        # # 消息传递层
        # for conv in self.convs:
        #     # for param in conv.state_dict():
        #     #     print(param)
        #     x_res = x
        #     x = conv(x, edge_index)
        #     x = F.relu(x)+x_res
        for i in range(self.num_layers):
            x_res = x
            x = self.convs1[i](x,edge_index)
            x = self.Dropout(x)
            x = self.convs2[i](x,edge_index)
            count += (x > 0).sum().item()

            x = F.relu(x)+x_res     
        # 全局池化 
        size1 = x.shape[0]*x.shape[1]           
        # activate_num += count / (x.shape[0]*x.shape[1]) / self.num_layers / 2
        x = global_add_pool(x, batch)
        
        # 分类器
        x = self.classifier1(x)
        x = self.Relu(x)
        size2 = x.shape[0]*x.shape[1]
        count += (x > 0).sum().item()
        #print(x.shape)
        
        activate_num+=count / (self.num_layers * size1 + size2)
        #print(activate_num)
        x = self.Dropout(x)
        x = self.classifier2(x)
        #self.convs.clear()
        #return self.classifier(x)
        return x.squeeze(),activate_num