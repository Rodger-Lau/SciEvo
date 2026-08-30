import os
import time
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from utils.utils import *
from utils.args import *
from utils.logging import *
from tools.gradient_compute import *
from tools.cluster import *
from torch.utils.tensorboard import SummaryWriter
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model

class Deep4NetWrapper(nn.Module):
    """Deep4Net"""
    def __init__(self, n_channels, n_classes, input_window_samples=1000):
        super(Deep4NetWrapper, self).__init__()
        
        # 使用braindecode的原始Deep4Net实现
        self.model = Deep4Net(
            n_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=2
        )
        
    def forward(self, x):
        # braindecode模型期望输入形状: (batch, channels, time)
        # 去掉额外的通道维度 (batch, 1, channels, time) -> (batch, channels, time)
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # 获取模型输出
        x = self.model(x)
        count = (x > 0).sum().item()
        size = 1
        for i in x.shape:
            size *= i
        activate_freq = count / size
        # 处理多时间点输出：如果是3D张量(batch, classes, time)，则对时间维度取平均
        if x.dim() == 3:
            x = x.mean(dim=-1)  # 对时间维度取平均，得到(batch, classes)
        
        return x, activate_freq

class ShallowFBCSPNetWrapper(nn.Module):
    """ShallowFBCSPNet包装器 - 严格按照论文实现"""
    def __init__(self, n_channels, n_classes, input_window_samples=1000):
        super(ShallowFBCSPNetWrapper, self).__init__()
        
        # 使用braindecode的原始ShallowFBCSPNet实现
        self.model = ShallowFBCSPNet(
            n_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=30
        )
        
        # 转换为密集预测模型
        to_dense_prediction_model(self.model)
        
    def forward(self, x):
        # braindecode模型期望输入形状: (batch, channels, time)
        # 去掉额外的通道维度 (batch, 1, channels, time) -> (batch, channels, time)
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # 获取模型输出
        x = self.model(x)
        
        # 处理多时间点输出：如果是3D张量(batch, classes, time)，则对时间维度取平均
        if x.dim() == 3:
            x = x.mean(dim=-1)  # 对时间维度取平均，得到(batch, classes)
        
        return x

class EEGNetPaper(nn.Module):
    """EEGNet模型  """
    def __init__(self, n_channels, n_classes, input_time_length=1000):
        super(EEGNetPaper, self).__init__()
        self.dropout = 0.1
        # 第一层: 时间卷积
        self.temporal_conv = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 第二层: 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(self.dropout)
        
        # 第三层: 可分离卷积
        self.separable_conv1 = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False)
        self.pointwise_conv1 = nn.Conv2d(32, 32, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(self.dropout)
        
        # 分类器
        self.classifier = nn.Linear(32 * ((input_time_length // 32)), n_classes)
        
    def forward(self, x):
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        # 第一层
        x = self.temporal_conv(x)
        x = self.bn1(x)
        
        # 第二层
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        # 第三层
        x = self.separable_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        # 分类
        x = x.view(x.size(0), -1)
        count = (x > 0).sum().item()
        size = 1
        for i in x.shape:
            size *= i
        activate_freq = count / size
        x = self.classifier(x)
        return x, activate_freq

class FixedTCN(nn.Module):
    """TCN模型"""
    def __init__(self, n_channels, n_classes, input_time_length=1000):
        super(FixedTCN, self).__init__()
        
        # 简化但稳定的TCN结构
        self.conv1 = nn.Conv1d(n_channels, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(16, 32, 5, padding=4, dilation=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(32, 64, 5, padding=8, dilation=4)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)
        
      
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 将任意长度序列池化为固定长度
        self.classifier = nn.Linear(64, n_classes)   # 标准分类器
        
    def forward(self, x):
        # 输入形状: (batch, 1, channels, time) -> 压缩为 (batch, channels, time)
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # TCN层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        
      
        x = self.global_pool(x)  # (batch, 64, time) -> (batch, 64, 1)
        x = x.squeeze(-1)       # (batch, 64, 1) -> (batch, 64)
        x = self.classifier(x)  # (batch, 64) -> (batch, n_classes)
        
        return x

class CompactCNNPaper(nn.Module):
    """CompactCNN模型 """
    def __init__(self, n_channels, n_classes, input_time_length=1000):
        super(CompactCNNPaper, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(1, 40, (1, 13), padding=(0, 6))
        self.bn1 = nn.BatchNorm2d(40)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(40, 40, (n_channels, 1), groups=40)
        self.bn2 = nn.BatchNorm2d(40)
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 3))
        self.dropout1 = nn.Dropout(0.5)
        
        # 第三层卷积
        self.conv3 = nn.Conv2d(40, 40, (1, 15), padding=(0, 7), groups=40)
        self.conv4 = nn.Conv2d(40, 40, 1)
        self.bn3 = nn.BatchNorm2d(40)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 3))
        self.dropout2 = nn.Dropout(0.5)
        
        # 分类器
        # 计算全连接层输入大小
        with torch.no_grad():
            x = torch.randn(1, 1, n_channels, input_time_length)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.avg_pool1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.avg_pool2(x)
            fc_input_size = x.view(1, -1).shape[1]
        
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_size, 80),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(80, n_classes)
        )
        
    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        # 第三层
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        # 分类
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MLPModel(nn.Module):
    """MLP模型 """
    def __init__(self, n_channels, n_classes, input_time_length=1000, hidden_dims=[1000, 500, 100]):
        super(MLPModel, self).__init__()
        
        input_dim = n_channels * input_time_length
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, n_classes))
        
        self.mlp = nn.Sequential(*layers)
        self.input_dim = input_dim
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.mlp(x)