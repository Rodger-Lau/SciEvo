import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np
import os
from util import config, file_dir
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from rdkit import Chem
# from rdkit.Chem import Draw, AllChem
from tqdm import tqdm
from utils.logging import *
from utils.utils import *
from models.GIN_robust import GIN
from tools.gradient_compute_BBBP import *
from tools.cluster import *
from model.DQN import *
from utils.args import *
import random
import copy
from torch.utils.tensorboard import SummaryWriter
import datetime
import pandas as pd
from collections import Counter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据处理类 - 从原始文件创建PyG数据集
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
weight_decay = 1e-4
dataset = 'BBBP'
writer_dir = f'Writer/BrainAI/{dataset}'
target_train_keep_ratios = [0.30]
use_curriculum_ordering = True
enable_freeze_editing = True
enable_ltp_ltd = True
ltp_freq_threshold = 0.528
ltd_freq_threshold = 0.516
max_ltp_events_per_run = 2
max_ltd_events_per_run = 2
bbbp_pos_weight_min = 0.2
bbbp_pos_weight_max = 5.0
bbbp_pos_weight_power = 0.5
bbbp_threshold_min = 0.2
bbbp_threshold_max = 0.85
bbbp_blend_bacc_weight = 0.25
bbbp_blend_acc_weight = 0.75
source_action_scale_map = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
target_action_scale_map = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
dqn_reward_auc_weight = 1.0
dqn_reward_bacc_weight = 0.75
dqn_reward_pred_gap_weight = 0.5
dqn_reward_loss_weight = 0.05
target_domain_init_epsilon = 0.20
target_domain_min_epsilon = 0.05
target_domain_explore_epochs = 15
target_checkpoint_min_bacc = 0.55
target_checkpoint_max_pred_pos = 0.95
non_freezable_layer_keywords = (
    'input_norm',
    'norms',
    'readout_norm',
    'classifier1',
    'classifier2',
)
if not os.path.exists(writer_dir):
    os.makedirs(writer_dir)
writer = SummaryWriter(writer_dir) 
exp_model = 'GIN_target_fewshot_30'
def get_config():
    parser = create_parser()
    args = parser.parse_args()
    
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    log_dir = f'logs/BrainAI/{dataset}/{exp_model}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(log_dir, __name__, '{}.log'.format(now))
    logger.info(args)
    
    return args, logger, now
args, logger, now = get_config()
logger.info(f'lr:{lr}, weight decay:{weight_decay}')
logger.info('Backbone patch active: RobustGIN(hidden=64, layers=2, mean+max+log_nodes readout, feature scaling, LayerNorm)')
logger.info('BBBP training patch active: independent exp state + best-val checkpoint restore + acc-leaning fixed/bacc/blend thresholds')
logger.info('DQN patch active: validation-aware reward + lighter freeze mapping + protected readout head')
logger.info(
    'Few-shot patch active: '
    f'target-train keep ratios={target_train_keep_ratios}, '
    'target val/test unchanged'
)
logger.info(
    'Module toggles: '
    f'curriculum={use_curriculum_ordering}, '
    f'freeze_editing={enable_freeze_editing}, '
    f'ltp_ltd={enable_ltp_ltd}'
)
logger.info(
    'Freeze guards: '
    f'non_freezable_keywords={non_freezable_layer_keywords}'
)
logger.info(
    'Memory thresholds: '
    f'ltp_freq>{ltp_freq_threshold}, ltd_freq<{ltd_freq_threshold}, '
    f'ltp_budget={max_ltp_events_per_run}, ltd_budget={max_ltd_events_per_run}'
)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def order_cluster_indices(cluster_members, sum_gradients_cluster):
    if use_curriculum_ordering:
        return sort_by_original_gradient(sum_gradients=sum_gradients_cluster)
    return sorted(cluster_members.keys())


def order_domains_within_cluster(cluster, sum_gradients):
    if use_curriculum_ordering:
        temp_sorted_gradients_list = sort_by_original_gradient(
            sum_gradients=[sum_gradients[domain] for domain in cluster]
        )
        return [cluster[index] for index in temp_sorted_gradients_list]
    return sorted(cluster)


def stratified_three_way_split(indices, labels, seed):
    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=labels
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_y,
        test_size=1 / 3,
        random_state=seed,
        stratify=temp_y
    )
    return list(train_idx), list(val_idx), list(test_idx)


def repeat_to_length(indices, target_length):
    if not indices:
        return []
    repeated = list(indices)
    ptr = 0
    while len(repeated) < target_length:
        repeated.append(indices[ptr % len(indices)])
        ptr += 1
    return repeated[:target_length]


def stratified_subsample_indices(indices, dataset_like, keep_ratio, seed):
    indices = list(indices)
    if keep_ratio >= 1.0 or len(indices) <= 1:
        return indices

    grouped = {}
    for idx in indices:
        label = int(dataset_like[idx].y.view(-1)[0].item())
        grouped.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    sampled = []
    for _, group in grouped.items():
        local_group = list(group)
        rng.shuffle(local_group)
        keep_n = max(1, int(round(len(local_group) * keep_ratio)))
        keep_n = min(len(local_group), keep_n)
        sampled.extend(local_group[:keep_n])

    rng.shuffle(sampled)
    return sampled


def _dataset_label_counts(dataset_like):
    labels = []
    for idx in range(len(dataset_like)):
        data = dataset_like[idx]
        labels.append(int(data.y.view(-1)[0].item()))
    counts = Counter(labels)
    return counts.get(0, 0), counts.get(1, 0)


def make_balanced_bce_criterion(dataset_like, device, logger=None, prefix=''):
    neg, pos = _dataset_label_counts(dataset_like)
    if neg == 0 or pos == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = (neg / pos) ** bbbp_pos_weight_power
        pos_weight_value = min(max(pos_weight_value, bbbp_pos_weight_min), bbbp_pos_weight_max)

    if logger is not None:
        total = neg + pos
        pos_ratio = pos / total if total else 0.0
        logger.info(
            f'{prefix} balanced BCE: neg={neg}, pos={pos}, '
            f'pos_ratio={pos_ratio:.4f}, pos_weight={pos_weight_value:.4f}, '
            f'power={bbbp_pos_weight_power:.2f}'
        )

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class BBBPDataset(InMemoryDataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        self.csv_file = csv_file
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_file)]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def _has_carbonyl(self, mol):
        """检查分子中是否含有羰基(C=O)"""
        pattern = Chem.MolFromSmarts('[CX3]=[OX1]')
        return mol.HasSubstructMatch(pattern)
    
    def _has_halogen(self, mol):
        """检查分子中是否含有卤素原子(F, Cl, Br, I)"""
        pattern = Chem.MolFromSmarts('[F,Cl,Br,I]')
        return mol.HasSubstructMatch(pattern)
    
    def process(self):
        df = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]))
        data_list = []
        
        # 统计各域的数量
        domain_counts = {
            (0, 0): 0,  # 无羰基、无卤素
            (0, 1): 0,  # 无羰基、有卤素
            (1, 0): 0,  # 有羰基、无卤素
            (1, 1): 0   # 有羰基、有卤素
        }
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Molecules"):
            smiles = row['smiles']
            label = row['p_np']
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                continue
                
            # 添加氢原子
            mol = Chem.AddHs(mol)
            
            # 确定分子所属的域
            has_carbonyl = int(self._has_carbonyl(mol))
            has_halogen = int(self._has_halogen(mol))
            domain = (has_carbonyl, has_halogen)
            domain_counts[domain] += 1
            
            # 获取原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),  # 原子序数
                    atom.GetDegree(),     # 度数
                    atom.GetImplicitValence(),  # 隐式价
                    int(atom.GetIsAromatic()),  # 是否芳香
                    atom.GetTotalNumHs()   # 连接的氢原子数
                ]
                atom_features.append(features)
            
            # 创建边索引
            edge_index = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                # 添加双向边
                edge_index.append([i, j])
                edge_index.append([j, i])
            
            # 转换为张量
            try:
                x = torch.tensor(atom_features, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                y = torch.tensor([label], dtype=torch.float)
                
                # 创建图数据对象，添加域信息
                data = Data(
                    x=x, 
                    edge_index=edge_index, 
                    y=y,
                    domain=torch.tensor([has_carbonyl, has_halogen], dtype=torch.long)
                )
                data.has_carbonyl = has_carbonyl
                data.has_halogen = has_halogen
                data_list.append(data)
            except Exception as e:
                print(f"Error processing SMILES: {smiles}, error: {e}")
                continue
        
        # 打印域统计信息
        logger.info("\nDomain Distribution:")
        logger.info(f"Domain (0,0): {domain_counts[(0,0)]} molecules")
        logger.info(f"Domain (0,1): {domain_counts[(0,1)]} molecules")
        logger.info(f"Domain (1,0): {domain_counts[(1,0)]} molecules")
        logger.info(f"Domain (1,1): {domain_counts[(1,1)]} molecules")
        
        # 保存处理后的数据
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved {len(data_list)} molecules to {self.processed_paths[0]}")



# 训练函数
def train(model, loader, optimizer, criterion, device, last_out, last_param, epoch, mode='acc'):
    noise_rate = 0.01
    last_out = last_out.to(device)
    #last_param.to(device)
    if mode == 'mape':
        last_param = 1-last_param
    last_param *= noise_rate
    model.train()
    #model.to(device)
    total_loss = 0
    correct = 0
    total = 0
    #loader.to(device)
    # bool1 = 2
    # bool2 = True
    activate_num = 0
    total_num = 0
    for data in loader:
        
        data.to(device)
        optimizer.zero_grad()
        # if bool1>0:
        #     print('传播前数据：')
        #     for name, param in model.named_parameters():
        #         print(name)
        #         print(param.data)
        #         break
        #     bool1-=1
        out, activate_num = model(data, activate_num)
        total_num += 1
        #loss = criterion(out, data.y)
        # print(last_out.device)
        # print(data.y.device)
        if last_out.shape != data.y.shape:
            last_out = data.y
        #target = last_param * out + (1-last_param) * data.y
        # print('out:',out)
        # print('target:',target)
        target = data.y
        #target = target.type(torch.long)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        # if bool2:
        #     print('更新后数据：')
        #     for name, param in model.named_parameters():
        #         print(name)
        #         print(param.data)
        #         break
        #     bool2 = False
        total_loss += loss.item() * data.num_graphs
        # pred = out.argmax(dim=1)
        # correct += (pred == data.y).sum().item()
        total += data.num_graphs
    return total_loss / total, activate_num / total_num

# 评估函数
@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    y_true, y_prob = [], []
    
    for data in loader:
        data = data.to(device)
        out,_ = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        
        # 收集预测结果
        y_true.append(data.y.cpu())
        y_prob.append(out.sigmoid().detach().cpu())
    
    y_true = torch.cat(y_true).view(-1).numpy()
    y_prob = torch.cat(y_prob).view(-1).numpy()
    
    # 将概率转换为类别预测
    y_pred = (y_prob >= threshold).astype(int)
    #print('真实值：',y_true,'预测值:', y_pred)
    # 计算各种评估指标
    results = {}
    results['label_pos_rate'] = float(np.mean(y_true)) if len(y_true) else 0.0
    results['prob_mean'] = float(np.mean(y_prob)) if len(y_prob) else 0.0
    results['pred_pos_rate'] = float(np.mean(y_pred)) if len(y_pred) else 0.0
    
    try:
        results['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        results['auc'] = 0.5
    
    try:
        results['accuracy'] = accuracy_score(y_true, y_pred)
    except:
        results['accuracy'] = 0.0

    try:
        results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    except:
        results['balanced_accuracy'] = 0.0
        
    try:
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
    except:
        results['precision'] = 0.0
        
    try:
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
    except:
        results['recall'] = 0.0
        
    try:
        results['f1'] = f1_score(y_true, y_pred)
    except:
        results['f1'] = 0.0
    
    # 计算混淆矩阵
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results['confusion_matrix'] = {
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp
        }
    except:
        results['confusion_matrix'] = None

    results['_y_true'] = y_true
    results['_y_prob'] = y_prob
    
    return total_loss / len(loader.dataset), results, out


def score_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            'balanced_accuracy': 0.0,
            'accuracy': 0.0,
            'f1': 0.0,
            'pred_pos_rate': 0.0,
            'label_pos_rate': float(np.mean(y_true)) if len(y_true) > 0 else 0.0,
        }

    y_pred = (y_prob >= threshold).astype(int)
    return {
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'pred_pos_rate': float(np.mean(y_pred)),
        'label_pos_rate': float(np.mean(y_true)),
    }


def threshold_candidates(y_prob):
    y_prob = np.asarray(y_prob).reshape(-1)
    if len(y_prob) == 0:
        return np.array([0.5])

    unique_probs = np.unique(np.clip(y_prob, 0.0, 1.0))
    if len(unique_probs) > 1:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
    else:
        midpoints = unique_probs
    candidates = np.unique(np.concatenate((
        [bbbp_threshold_min, 0.5, bbbp_threshold_max],
        unique_probs,
        midpoints,
    )))
    candidates = candidates[
        (candidates >= bbbp_threshold_min) &
        (candidates <= bbbp_threshold_max)
    ]
    if len(candidates) == 0:
        candidates = np.array([0.5])
    return candidates


def select_threshold_by_val(y_true, y_prob, mode='blend'):
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, score_threshold(y_true, y_prob, 0.5)

    best_threshold = 0.5
    best_info = None
    best_key = None
    for threshold in threshold_candidates(y_prob):
        info = score_threshold(y_true, y_prob, threshold)
        bacc = info['balanced_accuracy']
        acc = info['accuracy']
        f1 = info['f1']
        pred_gap = abs(info['pred_pos_rate'] - info['label_pos_rate'])
        if mode == 'bacc':
            score = bacc
        elif mode == 'acc':
            score = acc
        else:
            score = (
                bbbp_blend_bacc_weight * bacc +
                bbbp_blend_acc_weight * acc
            )
        key = (
            round(float(score), 12),
            round(float(bacc), 12),
            round(float(acc), 12),
            round(float(f1), 12),
            -pred_gap,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_info = dict(info)
            best_info['score'] = float(score)
            best_info['mode'] = mode

    return best_threshold, best_info


def select_thresholds_by_val(y_true, y_prob):
    fixed_threshold = 0.5
    bacc_threshold, bacc_info = select_threshold_by_val(y_true, y_prob, mode='bacc')
    blend_threshold, blend_info = select_threshold_by_val(y_true, y_prob, mode='blend')
    return {
        'fixed': (fixed_threshold, score_threshold(y_true, y_prob, fixed_threshold)),
        'bacc': (bacc_threshold, bacc_info),
        'blend': (blend_threshold, blend_info),
    }


def compute_dqn_reward(train_loss, val_metrics):
    val_auc = float(val_metrics.get('auc', 0.0))
    val_bacc = float(val_metrics.get('balanced_accuracy', 0.0))
    pred_pos_rate = float(val_metrics.get('pred_pos_rate', 0.0))
    label_pos_rate = float(val_metrics.get('label_pos_rate', 0.0))
    pred_gap = abs(pred_pos_rate - label_pos_rate)
    reward = (
        dqn_reward_auc_weight * val_auc +
        dqn_reward_bacc_weight * val_bacc -
        dqn_reward_pred_gap_weight * pred_gap -
        dqn_reward_loss_weight * float(train_loss)
    )
    return float(reward)


def is_non_freezable_layer(layer_name):
    return any(keyword in layer_name for keyword in non_freezable_layer_keywords)


def get_target_domain_epsilon(epoch_idx):
    if target_domain_explore_epochs <= 1:
        return target_domain_min_epsilon
    clipped_epoch = min(max(int(epoch_idx), 0), target_domain_explore_epochs - 1)
    progress = clipped_epoch / float(target_domain_explore_epochs - 1)
    return (
        target_domain_init_epsilon +
        (target_domain_min_epsilon - target_domain_init_epsilon) * progress
    )

# 打印评估结果的函数
def print_evaluation_metrics(metrics, prefix=""):
    print(f"{prefix}AUC: {metrics['auc']:.4f}")
    print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}Balanced Accuracy: {metrics.get('balanced_accuracy', 0.0):.4f}")
    print(f"{prefix}Pred Pos Rate: {metrics.get('pred_pos_rate', 0.0):.4f}")
    print(f"{prefix}Label Pos Rate: {metrics.get('label_pos_rate', 0.0):.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall: {metrics['recall']:.4f}")
    print(f"{prefix}F1 Score: {metrics['f1']:.4f}")
    
    if metrics['confusion_matrix']:
        cm = metrics['confusion_matrix']
        print(f"{prefix}Confusion Matrix:")
        print(f"{prefix}  True Positives: {cm['true_positive']}")
        print(f"{prefix}  True Negatives: {cm['true_negative']}")
        print(f"{prefix}  False Positives: {cm['false_positive']}")
        print(f"{prefix}  False Negatives: {cm['false_negative']}")
def compute_domain_node_statistics(data_list, device='cpu'):

    domain_x = {0: [], 1: [], 2: [], 3: []}

    for data in data_list:
        has_carbonyl = int(data.has_carbonyl.item())
        has_halogen = int(data.has_halogen.item())

        # 确定域编号
        if has_carbonyl and has_halogen:
            domain_id = 0
        elif has_carbonyl and not has_halogen:
            domain_id = 1
        elif not has_carbonyl and has_halogen:
            domain_id = 2
        else:
            domain_id = 3

        # data.x shape: (num_nodes, 7)
        domain_x[domain_id].append(data.x.to(device))

    # 计算每个域的均值和标准差
    cur_means = []
    cur_stds = []

    domain_names = [
        "Carbonyl & Halogen",
        "Carbonyl only",
        "Halogen only",
        "Neither"
    ]

    for domain_id in range(4):
        if len(domain_x[domain_id]) == 0:
            # 该域无数据，用零向量占位
            cur_means.append(torch.zeros(7, device=device))
            cur_stds.append(torch.zeros(7, device=device))
            print(f"  Domain {domain_id} ({domain_names[domain_id]}): 无数据")
            continue

        # 拼接该域所有图的所有节点特征 → (N_domain, 7)
        all_x = torch.cat(domain_x[domain_id], dim=0)

        # 沿节点维度统计
        mean = all_x.mean(dim=0)  # (7,)
        std = all_x.std(dim=0)    # (7,)

        cur_means.append(mean)
        cur_stds.append(std)

        print(f"  Domain {domain_id} ({domain_names[domain_id]:>20s}): "
            f"{len(domain_x[domain_id]):>3d} 图, "
              f"{all_x.shape[0]:>5d} 节点, "
              f"‖mean‖={mean}, "
              f"‖std‖={std}")

    return cur_means, cur_stds
def train_model_RL(model, train_loader, val_loader, test_loader, logger, cur_mean, cur_std, dqn_agent, action_scale_map, H_matrix, 
                   rl_optimizer, batch_size, c, write=False, 
                   common=True, n_epochs=50, new_domain=False, acc=0):
    model = model.to(device)
    rho = 0.001 * acc
    old_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    rel_changes = {}
    shift_changes = {}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = make_balanced_bce_criterion(
        train_loader.dataset,
        device,
        logger,
        prefix=f'[Domain {c}] RL'
    )
    current_state_vec = None
    activate_freq = 0.5
    activate_num = 0
    total_loss = 0
    total = 0
    best_score = -1.0
    best_select_key = None
    best_test_metrics = None
    best_model_state = None
    best_thresholds = select_thresholds_by_val([], [])
    best_test_metrics_by_threshold = {}
    cnt = 0
    early_stop = 10
    correct = 0
    ltp_events = 0
    ltd_events = 0
    for epoch in range(n_epochs):
        if common and write:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    writer.add_scalar(f'grad_norm/{name}', grad_norm, epoch)
        prev_params = {name: param.data.clone() for name, param in model.named_parameters()}
        if current_state_vec is None:
            model.eval()
            for param in model.parameters():
                param.requires_grad = True
            for data in train_loader:
        
                data.to(device)
                optimizer.zero_grad()
                out, _ = model(data, activate_num)
                target = data.y
                # if out.shape!=target.shape:
                #     out = out.unsqueeze(dim=0)
                #target = target.type(torch.long)
                loss = criterion(out, target)
                loss.backward()
                # optimizer.step()
                # total_loss += loss.item() * data.num_graphs
                pred = (out >= 0.5).cpu().numpy().astype(float)
                #pred = out.argmax(dim=1)
                break
            grad_flat = get_flat_grad(model).detach()
        else:
            # current_state_vec = state
            grad_flat = current_state_vec[0]
        grad_input = grad_flat.unsqueeze(0) # [1, total_params]
        mean_input = cur_mean.unsqueeze(0)  # [1, 11]
        std_input = cur_std.unsqueeze(0)
        logger.info(f'grad_input shape:{grad_input.shape}, mean_input shape:{mean_input.shape}, std_input shape:{std_input.shape}')
        current_state_vec = (grad_input, mean_input, std_input) 
        target_explore_prob = 0.0
        if new_domain:
            target_explore_prob = get_target_domain_epsilon(epoch)
        if new_domain and np.random.rand() <= target_explore_prob:
            logger.info(f'random choose (target epsilon={target_explore_prob:.4f})')
            action_idx = random.randrange(len(target_action_scale_map))
        else:
            action_idx = dqn_agent.select_action(current_state_vec, new_domain)
        current_state_vec = (grad_input.squeeze(0), mean_input.squeeze(0), std_input.squeeze(0)) 
        scale_map = target_action_scale_map if new_domain else action_scale_map
        scale = scale_map[action_idx]
        if not new_domain:
            logger.info(f'activate freq:{activate_freq}')
            if enable_ltp_ltd and activate_freq > ltp_freq_threshold and ltp_events < max_ltp_events_per_run:
                logger.info('高激活频率，LTP')
                scale *= 0.9
                ltp_events += 1
            elif enable_ltp_ltd and activate_freq < ltd_freq_threshold and ltd_events < max_ltd_events_per_run:
                logger.info('低激活频率，LTD')
                scale /= 0.9
                ltd_events += 1
        logger.info(f'ltp/ltd events:{ltp_events}/{max_ltp_events_per_run}, {ltd_events}/{max_ltd_events_per_run}')
        sorted_layers = sorted(H_matrix.items(), key=lambda item: item[1])
        candidate_layers = [
            (name, value) for name, value in sorted_layers
            if not is_non_freezable_layer(name)
        ]
        total_layers = len(sorted_layers)
        candidate_layer_count = len(candidate_layers)
        freeze_count = int(candidate_layer_count * scale) if enable_freeze_editing else 0
        layers_to_freeze = [name for name, _ in candidate_layers[:freeze_count]]
        logger.info(
            f'freeze scale:{scale} (new_domain={new_domain}) '
            f'candidate_layers={candidate_layer_count}/{total_layers} '
            f'freeze_count={freeze_count}'
        )

        # 训练模型
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if name in layers_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        activate_num = 0
        batch_num = 0
        for data in train_loader:  # 正常训练
    
            data.to(device)
            optimizer.zero_grad()
            out, activate_num = model(data, activate_num)
            if out.shape != data.y.shape:
                out = data.y
            target = (1 - rho) * data.y + rho * out
            # target = data.y
            # if out.shape!=target.shape:
            #     out = out.unsqueeze(dim=0)
            #target = target.type(torch.long)
            # logger.info(f'out shape:{out.shape}, target shape:{target.shape}')
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            batch_num += 1
            total_loss += loss.item() * data.num_graphs
            pred = (out >= 0.5).cpu().numpy().astype(float)
            #pred = out.argmax(dim=1)
            correct += (pred == data.y.cpu().numpy()).sum().item()
            total += data.num_graphs
        total_loss /= len(train_loader)
        activate_freq = activate_num / batch_num
        next_state_vec = copy.deepcopy(current_state_vec)
        updates = calculate_layer_updates(model, prev_params)
        for name, update_val in updates.items():
            if name not in layers_to_freeze:
                H_matrix[name] = 0.9 * H_matrix[name] + 0.1 * update_val # 平滑更新
        if not new_domain:
            model.eval()
            for param in model.parameters():
                param.requires_grad = True
            for data in train_loader:
        
                data.to(device)
                optimizer.zero_grad()
                out, _ = model(data, activate_num)
                target = data.y
                # if out.shape!=target.shape:
                #     out = out.unsqueeze(dim=0)
                #target = target.type(torch.long)
                loss = criterion(out, target)
                loss.backward()
                # optimizer.step()
                # total_loss += loss.item() * data.num_graphs
                pred = (out >= 0.5).cpu().numpy().astype(float)
                #pred = out.argmax(dim=1)
                break
            grad_flat = get_flat_grad(model).detach() # [total_params] 
            # logger.info(f'grad_flat shape:{grad_flat.shape}') # mean_grad_flats shape:torch.Size([1085096])
            next_state_vec = (grad_flat, cur_mean, cur_std)
            pass
        # test_loss, test_acc = test(model, test_loader, criterion, device)
        val_loss, val_metrics, val_output = evaluate(model, val_loader, criterion, device)
        val_auc = val_metrics['auc']
        val_acc = val_metrics['accuracy']
        if not new_domain:
            reward = compute_dqn_reward(total_loss, val_metrics)
            dqn_agent.memory.append((current_state_vec, action_idx, reward, next_state_vec))
            dqn_agent.train_step(batch_size=batch_size, optimizer=rl_optimizer)
            current_state_vec = copy.deepcopy(next_state_vec)
            logger.info(
                f"dqn reward={reward:.4f} "
                f"(val_auc={val_auc:.4f}, "
                f"val_bacc={val_metrics.get('balanced_accuracy', 0.0):.4f}, "
                f"pred_gap={abs(val_metrics.get('pred_pos_rate', 0.0) - val_metrics.get('label_pos_rate', 0.0)):.4f})"
            )
        constraint_satisfied = True
        if new_domain:
            constraint_satisfied = (
                val_metrics.get('balanced_accuracy', 0.0) >= target_checkpoint_min_bacc and
                val_metrics.get('pred_pos_rate', 1.0) <= target_checkpoint_max_pred_pos
            )
        select_key = (
            int(constraint_satisfied),
            round(float(val_auc), 12),
            round(float(val_metrics.get('balanced_accuracy', 0.0)), 12),
            -abs(val_metrics.get('pred_pos_rate', 0.0) - val_metrics.get('label_pos_rate', 0.0)),
        )
        if best_select_key is None or select_key > best_select_key:
            best_select_key = select_key
            best_score = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            best_thresholds = select_thresholds_by_val(
                val_metrics['_y_true'],
                val_metrics['_y_prob']
            )
            cnt = 0
            best_test_metrics_by_threshold = {}
            for threshold_name, (threshold_value, threshold_info) in best_thresholds.items():
                test_loss, test_metrics, test_output = evaluate(
                    model,
                    test_loader,
                    criterion,
                    device,
                    threshold=threshold_value
                )
                test_metrics = dict(test_metrics)
                test_metrics['threshold'] = float(threshold_value)
                test_metrics['val_threshold_info'] = dict(threshold_info)
                best_test_metrics_by_threshold[threshold_name] = test_metrics
            best_test_metrics = best_test_metrics_by_threshold['blend']
        else:
            cnt += 1
        blend_threshold = best_thresholds.get('blend', (0.5, {}))[0]
        bacc_threshold = best_thresholds.get('bacc', (0.5, {}))[0]
        logger.info(f"  [Domain {c:3d}]"
                f"Ep {epoch+1:3d}/{n_epochs}  "
                f"val_auc={val_auc:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"val_bacc={val_metrics.get('balanced_accuracy', 0.0):.4f}  "
                f"pred_pos={val_metrics.get('pred_pos_rate', 0.0):.4f}  "
                f"ckpt_ok={int(constraint_satisfied)}  "
                f"thr_blend={blend_threshold:.4f}  "
                f"thr_bacc={bacc_threshold:.4f}  "
                f"best_auc={best_score:.4f}  "
                f"patience={cnt}/{early_stop}")

        if cnt >= early_stop:
            logger.info(f"  [Domain {c:3d}]"
                    f"→ Early stop at epoch {epoch+1}")
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_test_metrics is None:
        test_loss, best_test_metrics, test_output = evaluate(
            model,
            test_loader,
            criterion,
            device,
            threshold=0.5
        )
        best_test_metrics = dict(best_test_metrics)
        best_test_metrics['threshold'] = 0.5
        best_test_metrics['val_threshold_info'] = {}
        best_test_metrics_by_threshold = {
            'fixed': best_test_metrics,
            'bacc': best_test_metrics,
            'blend': best_test_metrics,
        }
    test_auc = best_test_metrics['auc']
    test_acc = best_test_metrics['accuracy']
    fixed_metrics = best_test_metrics_by_threshold.get('fixed', {})
    bacc_metrics = best_test_metrics_by_threshold.get('bacc', {})
    blend_val_info = best_test_metrics.get('val_threshold_info', {})
    logger.info(f"\n  >>> [Domain {c:3d}] "
            f"AUC={test_auc:.4f}  ACC={test_acc:.4f}  "
            f"BACC={best_test_metrics.get('balanced_accuracy', 0.0):.4f}  "
            f"PRED_POS={best_test_metrics.get('pred_pos_rate', 0.0):.4f}  "
            f"THR_BLEND={best_test_metrics.get('threshold', 0.5):.4f}  "
            f"VAL_BLEND_BACC={blend_val_info.get('balanced_accuracy', 0.0):.4f}  "
            f"VAL_BLEND_ACC={blend_val_info.get('accuracy', 0.0):.4f}")
    logger.info(f"      threshold diagnostics: "
            f"fixed(ACC={fixed_metrics.get('accuracy', 0.0):.4f}, "
            f"BACC={fixed_metrics.get('balanced_accuracy', 0.0):.4f}, "
            f"PRED_POS={fixed_metrics.get('pred_pos_rate', 0.0):.4f}, "
            f"THR={fixed_metrics.get('threshold', 0.5):.4f})  "
            f"bacc(ACC={bacc_metrics.get('accuracy', 0.0):.4f}, "
            f"BACC={bacc_metrics.get('balanced_accuracy', 0.0):.4f}, "
            f"PRED_POS={bacc_metrics.get('pred_pos_rate', 0.0):.4f}, "
            f"THR={bacc_metrics.get('threshold', 0.5):.4f})")
    return test_auc, test_acc, model

def main():

    setup_seed(42)
    p0 = 0.5
    lambda0 = 1e-4
    batch_size = 8
    # 加载数据集
    data_dir = './data/bbbp'
    src_csv = './data/bbbp/BBBP.csv'
    dataset = BBBPDataset(root=data_dir, csv_file=src_csv)
    # dataset = load_mutag_dataset(cache_path='./data/MUTAG/mutag_data_list.pt',raw_root='./data/MUTAG')
    logger.info(f'数据集包含 {len(dataset)} 个图')
    domains = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    domains={}
    for i in range(4):
        domains[i]=[]
    sum_gradients = []
    
    # 收集每个域中的分子索引
    for idx in range(len(dataset)):
        data = dataset[idx]
        #print(data.has_aromatic_ring.item())

        has_carbonyl = data.has_carbonyl.item()
        has_halogen = data.has_halogen.item()
        
        if has_carbonyl and has_halogen: #
            domains[0].append(idx) # "Carbonyl and Halogen"
        elif has_carbonyl and not has_halogen:
            domains[1].append(idx) # "Carbonyl without Halogen"
        elif not has_carbonyl and has_halogen:
            domains[2].append(idx) # "Halogen without Carbonyl"
        else:
            domains[3].append(idx) # "No Carbonyl or Halogen"
    max_length = max(len(domains[key]) for key in domains)
    cat_grad_dict = {}
    train_loaders, val_loaders, test_loaders = [],[],[]
    raw_train_split_indices = {}
    raw_val_split_indices = {}
    raw_test_split_indices = {}
    # 数量平衡
 
    logger.info("计算各域节点层面均值和标准差:")
    cur_means, cur_stds = compute_domain_node_statistics(list(dataset))
    # exit(0)
    for key in domains:
        raw_domain_indices = list(domains[key])
        raw_labels = [int(dataset[idx].y.item()) for idx in raw_domain_indices]
        raw_counts = Counter(raw_labels)
        logger.info(
            f'domain {key} raw label distribution: '
            f'neg={raw_counts.get(0, 0)}, pos={raw_counts.get(1, 0)}, total={len(raw_domain_indices)}'
        )

        train_indices, val_indices, test_indices = stratified_three_way_split(
            raw_domain_indices,
            raw_labels,
            seed=args.seed
        )
        raw_train_split_indices[key] = list(train_indices)
        raw_val_split_indices[key] = list(val_indices)
        raw_test_split_indices[key] = list(test_indices)
        train_indices = repeat_to_length(train_indices, int(0.7 * max_length))

        train_counts = Counter(int(dataset[idx].y.item()) for idx in train_indices)
        val_counts = Counter(int(dataset[idx].y.item()) for idx in val_indices)
        test_counts = Counter(int(dataset[idx].y.item()) for idx in test_indices)
        logger.info(
            f'domain {key} split label distribution: '
            f'train(neg={train_counts.get(0, 0)}, pos={train_counts.get(1, 0)}, total={len(train_indices)}), '
            f'val(neg={val_counts.get(0, 0)}, pos={val_counts.get(1, 0)}, total={len(val_indices)}), '
            f'test(neg={test_counts.get(0, 0)}, pos={test_counts.get(1, 0)}, total={len(test_indices)})'
        )

        train_dataset = dataset[train_indices]
        val_dataset = dataset[val_indices]
        test_dataset = dataset[test_indices]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
        for data in test_loader:
            output_shape = data.y.shape
            break
        continue
    exp_repeat = 10
    new_i = 3
    logger.info(f'new_i: {new_i}')
    sample_data = dataset[0]
    in_channels = sample_data.x.shape[1]  # 获取输入特征维度
    fewshot_results = {}
    for keep_ratio in target_train_keep_ratios:
        ratio_pct = int(round(keep_ratio * 100))
        logger.info("\n" + "#" * 120)
        logger.info(
            f'Few-shot target-train start: keep_ratio={keep_ratio:.2f} ({ratio_pct}%), '
            'target val/test unchanged'
        )
        logger.info("#" * 120)
        test_aucs = []
        test_accs = []
        for exp_idx in range(exp_repeat):
            exp_seed = args.seed + exp_idx
            setup_seed(exp_seed)
            sum_gradients = []
            cat_grad_dict = {}
            logger.info(
                f"\n{'='*60}  Target train {ratio_pct}% | Experiment {exp_idx+1}/{exp_repeat}  {'='*60}"
            )
            logger.info(f'Experiment seed: {exp_seed}, independent gradient/order state reset')
            for key in domains:
                if key == new_i: continue
                train_loader = train_loaders[key]
                val_loader = val_loaders[key]
                test_loader = test_loaders[key]
                drop_out = 0.35
                model = GIN(
                    in_channels=in_channels,
                    hidden_dim=64,
                    num_layers=2,
                    dropout=drop_out
                ).to(device)
                # model = GAT(in_channels=7, hidden_channels=64, output_dim=2, num_heads=8, 
                #                dropout=0.5, negative_slope=0.2,residual=True).to(device)
                
                # 损失函数和优化器
                #criterion = nn.CrossEntropyLoss()
                criterion = make_balanced_bce_criterion(
                    train_loader.dataset,
                    device,
                    logger,
                    prefix=f'[Stage 1 Domain {key}]'
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                
                # 训练循环
                #best_acc = 0
                logger.info(f'{key}开始训练...')
                for epoch in range(1, first_stage_num_epochs+1):
                    train_loss, _ = train(model, train_loader, optimizer, criterion, device, last_out=torch.zeros(output_shape), last_param=0.0, epoch=epoch)
                    #test_loss, test_acc = test(model, test_loader, criterion, device)
                    #val_loss, val_acc = validate(model, test_loader, criterion, device)
                    # if test_acc > best_acc:
                    #     best_acc = test_acc
                    
                    #if epoch % 20 == 0:
                    # print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    #         f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                sum_grad, cat_grad = compute_gradient(model=model, train_loader=train_loader, device=device, criterion=criterion)
                sum_gradients.append(sum_grad)
                logger.info(sum_gradients[key])
                cat_grad_dict[key] = cat_grad
                #print(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f}')
            n_clusters = 2
            cluster_labels, cluster_members = clustering_dict(n_clusters=n_clusters, data_dict=cat_grad_dict, algorithm='kmeans')  # 对梯度做聚类
            logger.info('聚类结果为：')
            cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_grad_dict)
            logger.info(cluster_grad)
            # for key1, value1 in cluster_grad.items():
            #     print(key1, value1)
            min_grad, min_i = get_min_gradient(sum_gradients=cluster_grad,new_i=-1)
            #print(min_grad,min_i)
            common_model = GIN(
                in_channels=in_channels,
                hidden_dim=64,
                num_layers=2,
                dropout=0.35
            ).to(device)

                
            # 损失函数和优化器
            criterion = None
            
   
            sum_gradients_cluster = [sum(cluster_grad[i]) for i in range(len(cluster_grad))]
            sorted_clusters_gradients_list = order_cluster_indices(cluster_members, sum_gradients_cluster)

            common_train_loss = []
            common_val_acc = []
            second_stage_circle = 1
            second_stage_num_epochs = 10
            for _ in range(second_stage_circle):
                for cluster_index in sorted_clusters_gradients_list:
                    train_number=1
                    #print(cluster_index)
                    cluster = cluster_members[cluster_index]  # cluster中含有domain
                    sorted_gradients_list = order_domains_within_cluster(cluster, sum_gradients)
                    logger.info(sorted_gradients_list)
                    for key in sorted_gradients_list:
                        if key == new_i : continue
                        train_number-=1
                        if train_number>=0:
                            # local_dataset = dataset[domains[key]]
                            # weight_decay = my_sigmoid(lambda0,sum_gradients[key], sum_gradients[sorted_gradients_list[-1]])
                            optimizer = torch.optim.Adam(common_model.parameters(), lr=lr, weight_decay=weight_decay)
                            #print(key)
                            common_model.drop_out = my_sigmoid(p0,sum_gradients_cluster[cluster_index], sum_gradients_cluster[sorted_clusters_gradients_list[-1]])
                            #print(sum_gradients[key],sum_gradients[sorted_gradients_list[-1]])
                            train_loader = train_loaders[key]
                            val_loader = val_loaders[key]
                            test_loader = test_loaders[key]
                            criterion = make_balanced_bce_criterion(
                                train_loader.dataset,
                                device,
                                logger,
                                prefix=f'[Stage 2 Domain {key}]'
                            )
                            # 训练循环
                            test_acc = 0.0
                            best_acc = 0.0
                            best_auc = 0.5
                            last_output = torch.zeros(output_shape)
                            logger.info(f'{key}开始训练...')
                            
                            for epoch in range(1, second_stage_num_epochs+1):
                                train_loss, _ = train(common_model, train_loader, optimizer, criterion, device, last_out=torch.zeros(output_shape), last_param=0.0, epoch=epoch)
                                common_train_loss.append(train_loss)
                                val_loss, val_metrics, val_output = evaluate(common_model, val_loader, criterion, device)
                                test_loss, test_metrics, test_output = evaluate(common_model, test_loader, criterion, device)
                                val_acc = val_metrics['accuracy']
                                test_auc = test_metrics['auc']
                                common_val_acc.append(val_acc)
                            #     if test_acc > best_acc:
                            #         best_acc = test_acc
                            #     if test_auc > best_auc:
                            #         best_auc = test_auc
                                

            common_train_loss = []
            common_val_acc = []
            third_stage_circle = 3
            second_stage_num_epochs = 100
            total_params = len(cat_grad)
            action_scale_map = source_action_scale_map
            state_dim = 16
            in_dim = len(cur_means[0])
            dqn_agent = DQNAgent(grad_dim=total_params, encoding_dim=in_dim, data_dim=in_dim, state_dim=state_dim, action_dim=len(action_scale_map), device=device, logger=logger).to(device)
            rl_optimizer = optim.Adam(dqn_agent.q_net.parameters(), lr=0.001)
            H_matrix = {name: 0.0 for name, p in common_model.named_parameters()}
            for _ in range(third_stage_circle):
                for cluster_index in sorted_clusters_gradients_list:
                    train_number=1
                    #print(cluster_index)
                    cluster = cluster_members[cluster_index]  # cluster中含有domain
                    sorted_gradients_list = order_domains_within_cluster(cluster, sum_gradients)
                    logger.info(sorted_gradients_list)
                    for key in sorted_gradients_list:
                        if key == new_i : continue
                        train_number -= 1
                        if train_number >= 0:
                            # local_dataset = dataset[domains[key]]
                            # weight_decay = my_sigmoid(lambda0,sum_gradients[key], sum_gradients[sorted_gradients_list[-1]])
                            optimizer = torch.optim.Adam(common_model.parameters(), lr=lr, weight_decay=weight_decay)
                            #print(key)
                            common_model.drop_out = p0
                            #print(sum_gradients[key],sum_gradients[sorted_gradients_list[-1]])
                            train_loader = train_loaders[key]
                            val_loader = val_loaders[key]
                            test_loader = test_loaders[key]
                            # 训练循环
                            test_acc = 0.0
                            best_acc = 0.0
                            best_auc = 0.5
                            last_output = torch.zeros(output_shape)
                            logger.info(f'{key}开始训练...')
                            cur_mean = cur_means[key].to(device)
                            cur_std = cur_stds[key].to(device)
                            if True:
                                test_auc, test_acc, common_model = train_model_RL(common_model, train_loader, val_loader, test_loader, logger, cur_mean, cur_std, dqn_agent, action_scale_map, H_matrix, 
                                                                            rl_optimizer, batch_size, c=key, write=False, 
                                                                            common=True, n_epochs=second_stage_num_epochs, new_domain=False, acc=test_acc)
                            #     common_train_loss.append(train_loss)
                            #     #print('第',epoch, '轮激活频率为', activate_frequency)
                            #     test_loss, test_acc, last_output = test(common_model, val_loader, criterion, device)
    
                            #     #val_loss, val_acc = validate(common_model, test_loader, criterion, device)
                            #     val_loss, val_metrics, val_output = evaluate(common_model, val_loader, criterion, device)
                            #     test_loss, test_metrics, test_output = evaluate(common_model, test_loader,criterion, device)
                            #     val_acc = val_metrics['accuracy']
                            #     test_auc = test_metrics['auc']
                            #     common_val_acc.append(val_acc)
                            #     if test_acc > best_acc:
                            #         best_acc = test_acc
                            #     if test_auc > best_auc:
                            #         best_auc = test_auc
                                
                            #     if epoch % 20 == 0:
                            #         logger.info(f'第{epoch}轮激活频率为{activate_frequency}')
                            #         if activate_frequency > 0.01:
                            #             weight_decay *= 1.001
                            #             optimizer = torch.optim.Adam(common_model.parameters(), lr=0.01, weight_decay=weight_decay)
                            #             common_model.drop_out *= 1.001
                            #         elif activate_frequency<0.005:
                            #             weight_decay *=0.5
                            #             optimizer = torch.optim.Adam(common_model.parameters(), lr=0.01, weight_decay=weight_decay)
                            #             common_model.drop_out*=0.5
                            #         print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                            #             f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                            #             f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                            #         print_evaluation_metrics(val_metrics)
                            # logger.info(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
                            # print_evaluation_metrics(test_metrics)
            # exit(0)
            best_acc = 0
            coupled_train_loss = []
            coupled_val_acc = []
            # local_dataset = dataset[domains[new_i]]
            # train_indices = list(range(int(0.7*len(local_dataset))))
            # val_indices = list(range(int(0.7*len(local_dataset)),int(0.9*len(local_dataset))))
            # test_indices = list(range(int(0.9*len(local_dataset)), len(local_dataset)))
            # train_dataset = local_dataset[train_indices]
            # val_dataset = local_dataset[val_indices]
            # test_dataset = local_dataset[test_indices]
            # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            key = new_i
            target_train_indices = stratified_subsample_indices(
                raw_train_split_indices[new_i],
                dataset,
                keep_ratio=keep_ratio,
                seed=exp_seed
            )
            target_val_indices = raw_val_split_indices[new_i]
            target_test_indices = raw_test_split_indices[new_i]
            target_train_counts = Counter(int(dataset[idx].y.item()) for idx in target_train_indices)
            target_val_counts = Counter(int(dataset[idx].y.item()) for idx in target_val_indices)
            target_test_counts = Counter(int(dataset[idx].y.item()) for idx in target_test_indices)
            logger.info(
                f'[FewShot Target Domain {new_i}] keep_ratio={keep_ratio:.2f} ({ratio_pct}%), '
                f'train(neg={target_train_counts.get(0, 0)}, pos={target_train_counts.get(1, 0)}, total={len(target_train_indices)}), '
                f'val(neg={target_val_counts.get(0, 0)}, pos={target_val_counts.get(1, 0)}, total={len(target_val_indices)}), '
                f'test(neg={target_test_counts.get(0, 0)}, pos={target_test_counts.get(1, 0)}, total={len(target_test_indices)})'
            )
            train_dataset = dataset[target_train_indices]
            val_dataset = dataset[target_val_indices]
            test_dataset = dataset[target_test_indices]
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            fourth_stage_num_epochs = 200
            coupled_model = common_model
            coupled_model.to(device)
            coupled_learning_rate = lr
            coupled_weight_decay = weight_decay
            coupled_optimizer = torch.optim.Adam(coupled_model.parameters(), lr=coupled_learning_rate, weight_decay=coupled_weight_decay)
            cur_mean = cur_means[new_i].to(device)
            cur_std = cur_stds[new_i].to(device)
            test_auc, test_acc, common_model = train_model_RL(common_model, train_loader, val_loader, test_loader, logger, cur_mean, cur_std, dqn_agent, action_scale_map, H_matrix, 
                                                    rl_optimizer, batch_size, c=key, write=False, 
                                                    common=False, n_epochs=second_stage_num_epochs, new_domain=True, acc=test_acc)
            logger.info(f'第{exp_idx+1}次实验结果为： test_auc: {test_auc}, test_acc: {test_acc}')
            test_aucs.append(test_auc)
            test_accs.append(test_acc)
            # for epoch in range(1, fourth_stage_num_epochs+1):
            #     # def train(model, loader, optimizer, criterion, device, last_out, last_param, epoch, mode='acc')
            #     # train(new_sample_scaler, new_sample, epoch, coupled_optimizer, criterion, coupled_model, device, in_dim)
            #     # validate(new_sample_scaler,val_loader,criterion,coupled_model,device,in_dim)
            #     los, __, ___ = train(coupled_model,train_loader,coupled_optimizer,criterion,device,
            #                                                     last_param=1,last_out=torch.zeros(output_shape),epoch=epoch,mode='acc')
            #     coupled_train_loss.append(los)
            #def test(model, loader, criterion,device):
            # if True:
            #     val_loss, val_acc = validate(coupled_model, val_loader, criterion, device)
            #     coupled_val_acc.append(val_acc)
            #     test_loss, test_acc, test_out=test(coupled_model, test_loader, criterion, device)
            #     val_loss, val_metrics, val_output = evaluate(common_model, val_loader, criterion, device)
            #     test_loss, test_metrics, test_output = evaluate(common_model, test_loader,criterion, device)
            #     if test_acc > best_acc:
            #         best_acc = test_acc
            #     if test_auc > best_auc:
            #         best_auc = test_auc
            #     logger.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
            #                         f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
            #                         f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            #     print_evaluation_metrics(val_metrics)
                
            #     # if test_acc > best_acc:
            #     #         best_acc = test_acc
            # logger.info(f'最终测试准确率: {test_acc:.4f}, 最佳测试准确率: {best_acc:.4f},最终测试AUC: {test_auc:.4f}, 最佳测试AUC: {best_auc:.4f}')
            # print_evaluation_metrics(test_metrics)
                # 使用不同指标
            # with open("./metrics/mutag/common_train_loss.txt", 'w') as train_los:
            #     train_los.write(str(common_train_loss))
            # with open("./metrics/mutag/common_val_acc.txt", 'w') as validate_acc:
            #     validate_acc.write(str(common_val_acc))
            # with open("./metrics/mutag/coupled_train_loss.txt", 'w') as coupled_train_los:
            #     coupled_train_los.write(str(coupled_train_loss))
            # with open("./metrics/mutag/coupled_val_acc.txt", 'w') as coupled_validate_acc:
            #     coupled_validate_acc.write(str(coupled_val_acc))
        mean_auc = sum(test_aucs) / len(test_aucs)
        mean_acc = sum(test_accs) / len(test_accs)
        fewshot_results[keep_ratio] = {
            'aucs': list(test_aucs),
            'accs': list(test_accs),
            'mean_auc': mean_auc,
            'mean_acc': mean_acc,
        }
        logger.info("======================================================")
        logger.info(
            f"{exp_repeat}次随机顺序实验的 AUC (target train {ratio_pct}%): "
            f"{test_aucs}, 平均值: {round(mean_auc, 5)}"
        )
        logger.info(
            f"{exp_repeat}次随机顺序实验的 ACC (target train {ratio_pct}%): "
            f"{test_accs}, 平均值: {round(mean_acc, 5)}"
        )
        logger.info("======================================================")

    logger.info("\n" + "=" * 118)
    logger.info('Few-shot target-train summary (val/test unchanged)')
    for keep_ratio in target_train_keep_ratios:
        result = fewshot_results[keep_ratio]
        logger.info(
            f'keep_ratio={keep_ratio:.2f} ({int(round(keep_ratio * 100))}%): '
            f'mean_auc={result["mean_auc"]:.5f}, mean_acc={result["mean_acc"]:.5f}'
        )
    logger.info("=" * 118)
if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception('Exception')
