import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool
class GAT(nn.Module):
    def __init__(self,
                 in_channels,       # 输入特征维度
                 hidden_channels,   # 隐藏层维度
                 output_dim=1,      # 输出维度
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
        
        self.dropout = dropout
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
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_initial = x  # 保存初始特征用于残差连接
        
        # 第一层GAT
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # 使用ELU激活函数
        
        # 残差连接
        if self.residual and self.residual_linear is not None:
            res = self.residual_linear(x_initial)
            x = x + res
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        #x = global_add_pool(x, batch)
        # 第二层GAT
        x = self.conv2(x, edge_index)
        x = global_add_pool(x,batch)
        x = self.classifier1(x)
        x = F.relu(x)
        x = self.classifier2(x)
        #x= F.log_softmax(x, dim=1)
        # print(x.shape)
        # return F.log_softmax(x, dim=1)  # 输出log概率
        return  x.squeeze(),0

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