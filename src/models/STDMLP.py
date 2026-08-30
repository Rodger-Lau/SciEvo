import torch
import torch.nn as nn 
class StandardizedMLP(nn.Module):
    """标准化的MLP回归模型"""
    def __init__(self, input_channels=128, input_timesteps=100, output_timesteps=50):
        super(StandardizedMLP, self).__init__()
        
        self.input_channels = input_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        
        # 计算展平后的维度
        self.flat_input_dim = input_timesteps
        self.flat_output_dim = output_timesteps
        
        # 添加标志，只在第一次前向传播时打印形状
        self._first_forward = True
        
        # 模型结构
        self.flatten = nn.Flatten(start_dim=1)
        
        # 改进的网络结构 - 更深的网络
        self.fc1 = nn.Linear(self.flat_input_dim, 1024)
        self.bn = nn.BatchNorm1d(input_channels)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        
        self.fc7 = nn.Linear(32, self.flat_output_dim)
        
        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        """
        前向传播过程 - 只在第一次打印形状
        """
        # if self._first_forward and self.training:
        #     print(f"\n首次前向传播 - 形状变化:")
        #     print(f"  输入: {x.shape}")
        
        # 展平操作
        # x_flat = self.flatten(x)
        # if self._first_forward and self.training:
        #     print(f"  展平后: {x_flat.shape}")
        # x = x.transpose(1,2)
        # print('input x shape after transpose:', x.shape)
        # 全连接层1
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        # if self._first_forward and self.training:
        #     print(f"  全连接1后: {x.shape}")
        
        # 全连接层2
        x = self.fc2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        # if self._first_forward and self.training:
        #     print(f"  全连接2后: {x.shape}")
        
        # 全连接层3
        x = self.fc3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        # if self._first_forward and self.training:
        #     print(f"  全连接3后: {x.shape}")
        
        # 全连接层4
        x = self.fc4(x)
        x = self.bn(x)
        x = self.relu(x)
        # if self._first_forward and self.training:
        #     print(f"  全连接4后: {x.shape}")
        
        # 全连接层5
        x = self.fc5(x)
        x = self.bn(x)
        x = self.relu(x)
        # if self._first_forward and self.training:
        #     print(f"  全连接5后: {x.shape}")
        
        # 全连接层6
        x = self.fc6(x)
        x = self.bn(x)
        x = self.relu(x)
        # if self._first_forward and self.training:
        #     print(f"  全连接6后: {x.shape}")
        
        # 输出层
        x = self.fc7(x)
        # if self._first_forward and self.training:
            # print(f"  全连接7后: {x.shape}")
        
        # 重塑为原始形状
        # batch_size = x.shape[0]
        #output = output_flat.view(batch_size, self.input_channels, self.output_timesteps)
        
        # if self._first_forward and self.training:
        #     print(f"  重塑后: {x.shape}")
        #     self._first_forward = False
        # x = x.transpose(1,2)
        # print('outout x shape:', x.shape)
        return x