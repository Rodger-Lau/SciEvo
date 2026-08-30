import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from collections import deque, OrderedDict
import torch.nn.functional as F
# from utils.utils import StateEncoder
# ==========================================
# 3. DQN 智能体
# ==========================================
class StateEncoder(nn.Module):
    VALID_MODES = {'full', 'gradient_only', 'data_only'}

    def __init__(self, grad_dim, encoding_dim, data_dim, state_dim, state_encoding_mode='full'):
        """
        grad_dim: 模型参数总数量
        data_dim: 输入数据维度 (CIFAR-10为3通道，此处取3)
        state_dim: 最终状态向量维度
        """
        super(StateEncoder, self).__init__()
        if state_encoding_mode not in self.VALID_MODES:
            raise ValueError('state_encoding_mode must be full, gradient_only, or data_only')
        self.state_encoding_mode = state_encoding_mode
        # 将高维梯度降维到 data_dim (3)，以便与 mean, std 对齐
        self.fc_grad = None if state_encoding_mode == 'data_only' else nn.Linear(grad_dim, encoding_dim)
        
        # 可学习参数 W：将拼接后的向量 (mean, std, grad_proj) 映射到状态向量
        # 输入维度 = data_dim * 3
        if state_encoding_mode == 'full':
            state_input_dim = data_dim * 2 + encoding_dim
        elif state_encoding_mode == 'data_only':
            state_input_dim = data_dim * 2
        else:
            state_input_dim = encoding_dim
        self.W = nn.Linear(state_input_dim, state_dim)
        
    def forward(self, grad_flat, mean=None, std=None):
        """
        grad_flat: [batch, grad_dim]
        mean: [batch, data_dim]
        std: [batch, data_dim]
        """
        # full 使用数据统计与梯度；gradient_only 仅使用梯度；
        # data_only 仅使用 mean/std，且不实例化梯度投影层。
        if self.state_encoding_mode == 'full':
            if mean is None or std is None:
                raise ValueError('full state encoding requires mean and std')
            grad_proj = self.fc_grad(grad_flat) # [batch, data_dim]
            combined = torch.cat([mean, std, grad_proj], dim=1) # [batch, data_dim*3]
        elif self.state_encoding_mode == 'gradient_only':
            grad_proj = self.fc_grad(grad_flat)
            combined = grad_proj
        else:
            if mean is None or std is None:
                raise ValueError('data_only state encoding requires mean and std')
            combined = torch.cat([mean, std], dim=1)
        # print(f'combined shape:{combined.shape}') #grad_flat shape:torch.Size([16, 1, 1085096]), mean shape:torch.Size([16, 1, 11]), std shape:torch.Size([16, 1, 11]), combined shape:torch.Size([16, 3, 11])
        # grad_flat shape:torch.Size([16, 1085096]), mean shape:torch.Size([16, 11]), std shape:torch.Size([16, 11]),
        # combined shape:torch.Size([16, 33])
        # grad_flat shape:torch.Size([16, 1085096]), mean shape:torch.Size([16, 11]), std shape:torch.Size([16, 11]),
        # combined shape:torch.Size([16, 33])
        # 3. 导出低维状态 
        state = self.W(combined) # [batch, state_dim]
        return state
    

class ControllerNet(nn.Module):
    """
    DQN 的 Q 网络，用于预测“当前状态下选哪个单元最优”
    状态简单表示为当前网络已有的层数
    """
    def __init__(self, grad_dim, encoding_dim, data_dim, state_dim, action_dim, state_encoding_mode='full'):
        super(ControllerNet, self).__init__()
        self.state_encoding_mode = state_encoding_mode
        self.state_encoder = StateEncoder(grad_dim, encoding_dim, data_dim, state_dim, state_encoding_mode=state_encoding_mode)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
    def forward(self, state):
        if self.state_encoding_mode == 'full':
            grad_flat, mean, std = state
            x = self.state_encoder(grad_flat, mean, std)
        elif self.state_encoding_mode == 'gradient_only':
            grad_flat = state[0] if isinstance(state, (tuple, list)) else state
            x = self.state_encoder(grad_flat)
        else:
            if not isinstance(state, (tuple, list)) or len(state) < 3:
                raise ValueError('data_only controller state must contain gradient placeholder, mean, and std')
            grad_flat, mean, std = state
            x = self.state_encoder(grad_flat, mean, std)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DQNAgent(nn.Module):
    def __init__(self, grad_dim, encoding_dim, data_dim, state_dim, action_dim, device, logger=None, state_encoding_mode='full'):
        super(DQNAgent, self).__init__()
        if state_encoding_mode not in StateEncoder.VALID_MODES:
            raise ValueError('state_encoding_mode must be full, gradient_only, or data_only')
        self.state_encoding_mode = state_encoding_mode
        # self.fc1 = nn.Linear(state_dim, 64)
        # self.fc2 = nn.Linear(64, action_dim)
        self.q_net = ControllerNet(grad_dim=grad_dim, encoding_dim=encoding_dim, data_dim=data_dim, state_dim=state_dim, action_dim=action_dim, state_encoding_mode=state_encoding_mode).to(device)
        self.target_net = ControllerNet(grad_dim=grad_dim, encoding_dim=encoding_dim, data_dim=data_dim, state_dim=state_dim, action_dim=action_dim, state_encoding_mode=state_encoding_mode).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.epoch = 0
        self.logger = logger
        self.device = device
        if self.logger is not None:
            self.logger.info(f'DQN state encoding mode: {self.state_encoding_mode}')
    # def forward(self, x):
    #     x = torch.relu(self.fc1(x))
    #     return self.fc2(x)
    
    def select_action(self, state, new_domain=False):
        # Epsilon-Greedy 策略
        if np.random.rand() <= self.epsilon and not new_domain:
            self.logger.info('random choose')
            return random.randrange(9) # 0-8 对应 scale 0.1-0.9
        with torch.no_grad():
            self.logger.info('greedy choose')
            q_values = self.q_net(state)
            return q_values.argmax().item()

    def train_step(self, batch_size, optimizer):
        if len(self.memory) < batch_size:
            return
        self.epoch += 1
        minibatch = random.sample(self.memory, batch_size)
        # 这里简化DQN训练，实际应用中可用Target Network
        # 解包数据
        states, actions, rewards, next_states = zip(*minibatch)
        if self.state_encoding_mode == 'gradient_only':
            def gradient_state_tuple(state):
                gradient = state[0] if isinstance(state, (tuple, list)) else state
                placeholder = torch.empty(0)
                return gradient, placeholder, placeholder
            states = tuple(gradient_state_tuple(state) for state in states)
            next_states = tuple(gradient_state_tuple(state) for state in next_states)
        # self.logger.info(f'states:{states}')
        # grad_flat, mean, std = states
        # grad_flat = torch.stack(grad_flat).to(self.device)
        # mean = torch.stack(mean).to(self.device)
        # std = torch.stack(std).to(self.device)
        # state_tuples 是一个列表，包含 batch_size 个 元组
        # 我们先解构 state_tuples
        grads = [s[0] for s in states]
        means = [s[1] for s in states]
        stds = [s[2] for s in states]
        
        # 同理处理 next_state
        next_grads = [s[0] for s in next_states]
        next_means = [s[1] for s in next_states]
        next_stds = [s[2] for s in next_states]
        
        # 堆叠成 Batch 张量
        # 如果存储时已经存了 numpy，torch.stack 会自动转换
        grads_batch = torch.stack([torch.tensor(g) if not isinstance(g, torch.Tensor) else g for g in grads]).float().to(self.device)
        means_batch = torch.stack([torch.tensor(m) if not isinstance(m, torch.Tensor) else m for m in means]).float().to(self.device)
        stds_batch = torch.stack([torch.tensor(s) if not isinstance(s, torch.Tensor) else s for s in stds]).float().to(self.device)
        
        next_grads_batch = torch.stack([torch.tensor(g) if not isinstance(g, torch.Tensor) else g for g in next_grads]).float().to(self.device)
        next_means_batch = torch.stack([torch.tensor(m) if not isinstance(m, torch.Tensor) else m for m in next_means]).float().to(self.device)
        next_stds_batch = torch.stack([torch.tensor(s) if not isinstance(s, torch.Tensor) else s for s in next_stds]).float().to(self.device)
        # state = torch.tensor(state, dtype=torch.float32).to(self.device)
        states = (grads_batch, means_batch, stds_batch)
        # states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # next_states = torch.stack(next_states).to(self.device)
        next_states = (next_grads_batch, next_means_batch, next_stds_batch)
        
        # 计算当前Q
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if self.epoch % 5 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())  
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay