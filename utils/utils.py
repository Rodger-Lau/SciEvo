import numpy as np
import torch
import copy
import torch.nn as nn
def save_prompt_weights(model, file_path):
    torch.save(model.prompt.data, file_path)


def load_prompt_weights(model, file_path, device_name='cuda:0'):
    model.prompt.data = torch.load(file_path, map_location=device_name)
    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def calculate_variance(weight_snapshots):
    # Calculating variance across epochs for each weight matrix element-wise
    weight_snapshots = np.stack(weight_snapshots, axis=0)
    variance = np.var(weight_snapshots, axis=0)
    return variance

def get_tensor_size(tensor):
    size = 1
    for i in tensor.shape:
        size*=i
    return size
# ==========================================
# 2. 状态编码器
# ==========================================
class StateEncoder(nn.Module):
    def __init__(self, grad_dim, data_dim, state_dim):
        """
        grad_dim: 模型参数总数量
        data_dim: 输入数据维度 (CIFAR-10为3通道，此处取3)
        state_dim: 最终状态向量维度
        """
        super(StateEncoder, self).__init__()
        # 将高维梯度降维到 data_dim (3)，以便与 mean, std 对齐
        self.fc_grad = nn.Linear(grad_dim, data_dim)
        
        # 可学习参数 W：将拼接后的向量 (mean, std, grad_proj) 映射到状态向量
        # 输入维度 = data_dim * 3
        self.W = nn.Linear(data_dim * 3, state_dim)
        
    def forward(self, grad_flat, mean, std):
        """
        grad_flat: [batch, grad_dim]
        mean: [batch, data_dim]
        std: [batch, data_dim]
        """
        # 1. 梯度降维
        grad_proj = self.fc_grad(grad_flat) # [batch, data_dim]
        
        # 2. 拼接
        # 注意：mean和std可能需要unsqueeze增加batch维度
        combined = torch.cat([mean, std, grad_proj], dim=1) # [batch, data_dim*3]
        
        # 3. 导出低维状态
        state = self.W(combined) # [batch, state_dim]
        return state

# ==========================================
# 4. 辅助函数
# ==========================================
def get_flat_grad(model):
    """获取模型所有参数的梯度并展平"""
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)

def calculate_layer_updates(model, prev_params):
    """
    计算每一层参数的变化幅度 (L2范数)
    返回一个字典: {layer_name: update_norm}
    """
    updates = {}
    for name, param in model.named_parameters():
        if name in prev_params:
            # 计算参数变化量
            diff = param.data - prev_params[name]
            # 计算L2范数作为变化幅度
            updates[name] = torch.norm(diff).item() / get_tensor_size(diff)
    return updates

def freeze_layers(model, layer_names_to_freeze):
    """冻结指定层的参数"""
    for name, param in model.named_parameters():
        if name in layer_names_to_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True
# ==========================================
# 5. 核心逻辑：梯度屏蔽
# ==========================================
def mask_gradients_by_layer(model, layers_to_freeze):
    """
    将指定层的梯度置零，模拟冻结效果。
    这样在 optimizer.step() 时这些层不会更新，
    但之前 get_flat_grad() 获取到的梯度维度是完整的。
    """
    for name, param in model.named_parameters():
        if name in layers_to_freeze:
            if param.grad is not None:
                param.grad.data.zero_()


def get_data_stats(dataloader):
    """
    获取数据的 mean 和 std (针对每个batch)
    这里简化为计算一个batch的统计量
    """
    images, _ = next(iter(dataloader))
    # images shape: [B, C, H, W]
    # 计算每个通道的均值和方差
    # 维度 n=3 (channels)
    mean = images.mean(dim=(0, 2, 3)) # [3]
    std = images.std(dim=(0, 2, 3))   # [3]
    return mean, std


class MaskedMAELoss(torch.nn.Module):
    
    def __init__(self):
        super(MaskedMAELoss, self).__init__()

    def forward(self, v_, v):
        mask = (v != 0.0)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(v_ - v)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=30, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_checkpoint = None
        self.best_epoch = 0
        self.current_epoch = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss
        self.current_epoch += 1
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.best_checkpoint = copy.deepcopy(model.state_dict())
        self.best_epoch = self.current_epoch
        self.val_loss_min = val_loss

       