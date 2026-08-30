import os
import time
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import csv
import pickle
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')
_ORIGINAL_SYS_ARGV = sys.argv.copy()



from utils.utils import *
from utils.args import *
from utils.logging import *
from tools.gradient_compute import *
from tools.cluster import *
from torch.utils.tensorboard import SummaryWriter
from model.EEG_models import *
from model.DQN import *
dataset = 'CHBMIT'
writer_dir = f'Writer/BrainAI/{dataset}'
big_threshold = 0.8
small_threshold = 0.2
if not os.path.exists(writer_dir):
    os.makedirs(writer_dir)
writer = SummaryWriter(writer_dir)
def ensure_dir(path):
    if path is not None and path != "":
        os.makedirs(path, exist_ok=True)


def to_builtin(obj):
    """Convert numpy / torch objects to JSON-serializable builtin types."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, ensure_ascii=False, indent=2)


def save_pickle(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def append_csv_row(csv_path, fieldnames, row):
    ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer_csv.writeheader()
        writer_csv.writerow(to_builtin(row))


def make_result_dir(dataset_name, backbone_name, now_str):
    result_dir = os.path.join("results", "BrainAI", str(dataset_name), str(backbone_name), str(now_str))
    ensure_dir(result_dir)
    return result_dir


def init_history_dict():
    return {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_acc': 0.0,
        # For external layer-change visualization
        'layer_update_norms': {},
        'layer_relative_changes': {},
        # For external hippocampus / RL visualization
        'H_matrix_history': [],
        'frozen_layers_history': [],
        'rl_action_history': [],
        'activate_freq_history': []
    }


def record_layer_changes(history, model, prev_params, updates):
    if history is None:
        return
    history.setdefault('layer_update_norms', {})
    history.setdefault('layer_relative_changes', {})
    param_dict = dict(model.named_parameters())
    for name, update_val in updates.items():
        history['layer_update_norms'].setdefault(name, []).append(float(update_val))
        if name in prev_params and name in param_dict:
            old_param = prev_params[name].detach()
            new_param = param_dict[name].detach()
            delta = new_param - old_param
            rel_change = torch.norm(delta).item() / (torch.norm(old_param).item() + 1e-12)
            history['layer_relative_changes'].setdefault(name, []).append(float(rel_change))


def record_rl_state(history, epoch, H_matrix=None, layers_to_freeze=None,
                    action_idx=None, scale=None, freeze_count=None,
                    total_layers=None, activate_freq=None):
    if history is None:
        return
    history.setdefault('H_matrix_history', [])
    history.setdefault('frozen_layers_history', [])
    history.setdefault('rl_action_history', [])
    history.setdefault('activate_freq_history', [])

    if H_matrix is not None:
        history['H_matrix_history'].append({str(k): float(v) for k, v in H_matrix.items()})
    if layers_to_freeze is not None:
        history['frozen_layers_history'].append([str(x) for x in layers_to_freeze])

    history['rl_action_history'].append({
        'epoch': int(epoch) if epoch is not None else None,
        'action_idx': int(action_idx) if action_idx is not None else None,
        'freeze_scale': float(scale) if scale is not None else None,
        'freeze_count': int(freeze_count) if freeze_count is not None else None,
        'total_layers': int(total_layers) if total_layers is not None else None
    })
    if activate_freq is not None:
        history['activate_freq_history'].append(float(activate_freq))



def compute_chbmit_binary_metrics(cm, test_report=None, n_test_windows=None, stride_sec=2.0):
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        return {
            'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0,
            'precision': 0.0, 'recall': 0.0, 'sensitivity': 0.0,
            'specificity': 0.0, 'f1': 0.0, 'fp_per_hour_approx': 0.0
        }

    tn, fp, fn, tp = cm.ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if n_test_windows is None:
        n_test_windows = int(cm.sum())
    hours = (float(n_test_windows) * float(stride_sec)) / 3600.0
    fp_per_hour = fp / hours if hours > 0 else 0.0

    return {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1': float(f1),
        'fp_per_hour_approx': float(fp_per_hour),
    }


def save_experiment_results(result_dir, experiment_idx, dataset_name, backbone_name,
                            target_subject, model_name, test_acc, test_report,
                            cm, history, params, training_time, logger=None):
    """Save one experiment for later paper tables and external visualization."""
    exp_dir = os.path.join(result_dir, f"exp_{experiment_idx}")
    ensure_dir(exp_dir)

    save_json(history, os.path.join(exp_dir, "history.json"))
    save_pickle(history, os.path.join(exp_dir, "history.pkl"))
    save_json(test_report, os.path.join(exp_dir, "test_report.json"))

    if cm is not None:
        np.save(os.path.join(exp_dir, "confusion_matrix.npy"), cm)
        save_json(cm.tolist() if hasattr(cm, "tolist") else cm,
                  os.path.join(exp_dir, "confusion_matrix.json"))

    train_loss = history.get('train_loss', []) if isinstance(history, dict) else []
    val_loss = history.get('val_loss', []) if isinstance(history, dict) else []
    train_acc = history.get('train_acc', []) if isinstance(history, dict) else []
    val_acc = history.get('val_acc', []) if isinstance(history, dict) else []

    binary_metrics = compute_chbmit_binary_metrics(
        cm=cm,
        test_report=test_report,
        n_test_windows=int(np.asarray(cm).sum()) if cm is not None else None,
        stride_sec=2.0
    )

    exp_summary = {
        'experiment_idx': int(experiment_idx),
        'dataset': str(dataset_name),
        'backbone': str(backbone_name),
        'model_name': str(model_name),
        'target_subject': int(target_subject),
        'test_acc': float(test_acc),
        'precision': float(binary_metrics['precision']),
        'recall': float(binary_metrics['recall']),
        'sensitivity': float(binary_metrics['sensitivity']),
        'specificity': float(binary_metrics['specificity']),
        'f1': float(binary_metrics['f1']),
        'fp_per_hour_approx': float(binary_metrics['fp_per_hour_approx']),
        'tn': int(binary_metrics['tn']),
        'fp': int(binary_metrics['fp']),
        'fn': int(binary_metrics['fn']),
        'tp': int(binary_metrics['tp']),
        'best_val_acc': float(max(val_acc)) if len(val_acc) > 0 else 0.0,
        'final_train_loss': float(train_loss[-1]) if len(train_loss) > 0 else 0.0,
        'final_val_loss': float(val_loss[-1]) if len(val_loss) > 0 else 0.0,
        'final_train_acc': float(train_acc[-1]) if len(train_acc) > 0 else 0.0,
        'final_val_acc': float(val_acc[-1]) if len(val_acc) > 0 else 0.0,
        'params': int(params),
        'training_time': float(training_time)
    }

    save_json(exp_summary, os.path.join(exp_dir, "experiment_summary.json"))
    append_csv_row(
        os.path.join(result_dir, "summary.csv"),
        fieldnames=[
            'experiment_idx', 'dataset', 'backbone', 'model_name', 'target_subject',
            'test_acc', 'precision', 'recall', 'sensitivity', 'specificity', 'f1',
            'fp_per_hour_approx', 'tn', 'fp', 'fn', 'tp',
            'best_val_acc', 'final_train_loss', 'final_val_loss',
            'final_train_acc', 'final_val_acc', 'params', 'training_time'
        ],
        row=exp_summary
    )

    if logger is not None:
        logger.info(f"Saved experiment {experiment_idx} results to: {exp_dir}")
    return exp_summary


def save_overall_summary(result_dir, dataset_name, backbone_name, all_test_acc, logger=None):
    all_test_acc = [float(x) for x in all_test_acc]
    overall_summary = {
        'dataset': str(dataset_name),
        'backbone': str(backbone_name),
        'num_experiments': int(len(all_test_acc)),
        'all_test_acc': all_test_acc,
        'mean_acc': float(np.mean(all_test_acc)) if len(all_test_acc) > 0 else 0.0,
        'std_acc': float(np.std(all_test_acc)) if len(all_test_acc) > 0 else 0.0,
        'min_acc': float(np.min(all_test_acc)) if len(all_test_acc) > 0 else 0.0,
        'max_acc': float(np.max(all_test_acc)) if len(all_test_acc) > 0 else 0.0
    }
    save_json(overall_summary, os.path.join(result_dir, "overall_summary.json"))
    if logger is not None:
        logger.info(f"Saved overall summary to: {os.path.join(result_dir, 'overall_summary.json')}")
    return overall_summary


def _safe_list_get(values, idx, default=None):
    if values is None:
        return default
    if idx < 0 or idx >= len(values):
        return default
    return values[idx]


def _layer_epoch_snapshot(layer_dict, epoch_idx):
    """Extract one epoch from {layer_name: [v1, v2, ...]}."""
    if not isinstance(layer_dict, dict):
        return {}
    out = {}
    for name, values in layer_dict.items():
        if values is None or epoch_idx >= len(values):
            continue
        out[str(name)] = float(values[epoch_idx])
    return out


def build_learning_loop_entries(history, metadata):
    if history is None:
        return []

    rl_actions = history.get('rl_action_history', [])
    h_history = history.get('H_matrix_history', [])
    frozen_history = history.get('frozen_layers_history', [])
    activate_history = history.get('activate_freq_history', [])
    train_loss = history.get('train_loss', [])
    train_acc = history.get('train_acc', [])
    val_loss = history.get('val_loss', [])
    val_acc = history.get('val_acc', [])
    layer_updates = history.get('layer_update_norms', {})
    layer_rel_changes = history.get('layer_relative_changes', {})

    n_steps = max(
        len(rl_actions), len(h_history), len(frozen_history),
        len(activate_history), len(train_loss), len(train_acc),
        len(val_loss), len(val_acc)
    )

    entries = []
    for step_idx in range(n_steps):
        action_info = _safe_list_get(rl_actions, step_idx, {}) or {}
        frozen_layers = _safe_list_get(frozen_history, step_idx, []) or []

        entry = {}
        entry.update(metadata)
        entry.update({
            'local_rl_step': int(step_idx + 1),
            'epoch': int(action_info.get('epoch', step_idx + 1)) if action_info.get('epoch', step_idx + 1) is not None else None,
            'action_idx': action_info.get('action_idx', None),
            'freeze_scale': action_info.get('freeze_scale', None),
            'freeze_count': action_info.get('freeze_count', len(frozen_layers)),
            'total_layers': action_info.get('total_layers', None),
            'num_frozen_layers': int(len(frozen_layers)),
            'frozen_layers': [str(x) for x in frozen_layers],
            'activate_freq': _safe_list_get(activate_history, step_idx, None),
            'train_loss': _safe_list_get(train_loss, step_idx, None),
            'train_acc': _safe_list_get(train_acc, step_idx, None),
            'val_loss': _safe_list_get(val_loss, step_idx, None),
            'val_acc': _safe_list_get(val_acc, step_idx, None),
            # DQN reward in source-domain RL is train_acc in the original code.
            'reward_train_acc': _safe_list_get(train_acc, step_idx, None),
            'H_matrix': _safe_list_get(h_history, step_idx, {}) or {},
            'layer_update_norms': _layer_epoch_snapshot(layer_updates, step_idx),
            'layer_relative_changes': _layer_epoch_snapshot(layer_rel_changes, step_idx),
        })
        entries.append(entry)
    return entries


def save_learning_loop_trace(exp_dir, learning_loop_trace, target_learning_loop_trace=None, logger=None):
    ensure_dir(exp_dir)
    if target_learning_loop_trace is None:
        target_learning_loop_trace = []

    # Add global step ids after all source-domain calls are collected.
    for idx, item in enumerate(learning_loop_trace):
        item['global_rl_step'] = int(idx + 1)
    for idx, item in enumerate(target_learning_loop_trace):
        item['target_rl_step'] = int(idx + 1)

    save_json(learning_loop_trace, os.path.join(exp_dir, 'source_learning_loop_trace.json'))
    save_pickle(learning_loop_trace, os.path.join(exp_dir, 'source_learning_loop_trace.pkl'))
    save_json(target_learning_loop_trace, os.path.join(exp_dir, 'target_learning_loop_trace.json'))
    save_pickle(target_learning_loop_trace, os.path.join(exp_dir, 'target_learning_loop_trace.pkl'))

    csv_path = os.path.join(exp_dir, 'source_learning_loop_trace.csv')
    fieldnames = [
        'experiment_idx', 'phase', 'circle_index', 'cluster_index',
        'subject_index', 'subject', 'global_rl_step', 'local_rl_step', 'epoch',
        'action_idx', 'freeze_scale', 'freeze_count', 'total_layers',
        'num_frozen_layers', 'activate_freq', 'train_loss', 'train_acc',
        'val_loss', 'val_acc', 'reward_train_acc'
    ]
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for item in learning_loop_trace:
            row = {k: item.get(k, None) for k in fieldnames}
            writer_csv.writerow(to_builtin(row))

    if logger is not None:
        logger.info(f"Saved source learning-loop trace to: {os.path.join(exp_dir, 'source_learning_loop_trace.json')}")
        logger.info(f"Saved compact learning-loop CSV to: {csv_path}")

def get_config(backbone_name, dataset_for_log=None):
    parser = create_parser()
    args, _ = parser.parse_known_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if dataset_for_log is None:
        dataset_for_log = dataset
    log_dir = f'logs/BrainAI/{dataset_for_log}/{backbone_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(log_dir, __name__, '{}.log'.format(now))
    logger.info(args)

    return args, logger, now




def build_subject_to_cluster(cluster_members):
    subject_to_cluster = {}
    if isinstance(cluster_members, dict):
        iterable = cluster_members.items()
    else:
        iterable = enumerate(cluster_members)
    for cluster_idx, members in iterable:
        for subject_idx in members:
            subject_to_cluster[int(subject_idx)] = int(cluster_idx)
    return subject_to_cluster

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seed(2026)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
CLASS_NAMES = ['Non-seizure', 'Seizure']

def load_preprocessed_data(npz_path):
    print(f"加载数据: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    X_train = data['X_train'].astype(np.float32)
    y_train = data['y_train'].astype(np.int64)
    X_valid = data['X_valid'].astype(np.float32)
    y_valid = data['y_valid'].astype(np.int64)
    X_test = data['X_test'].astype(np.float32)
    y_test = data['y_test'].astype(np.int64)
    
    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"验证集: {X_valid.shape}, 标签: {y_valid.shape}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
    
    classes, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(classes, counts):
        percentage = count / len(y_train) * 100
        cls_name = CLASS_NAMES[cls]
        print(f"  {cls_name}({cls}): {count} ({percentage:.1f}%)")
    cur_mean = np.mean(X_train, axis=(0,2)) if len(X_train) > 0 else np.zeros(len(X_train[0]))
    cur_std = np.std(X_train, axis=(0,2)) if len(X_train) > 0 else np.zeros(len(X_train[0])) + (1e-8)
    # print(f'cur mean shape:{cur_mean.shape}; cur std shape:{cur_std.shape}')
    # print(f'X_train shape:{X_train.shape}') # X_train shape:(64, 44, 1125)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, cur_mean, cur_std

def create_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=32):
    # 添加通道维度 (batch, 1, channels, time)
    X_train = X_train[:, np.newaxis, :, :]
    X_valid = X_valid[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]
    
    # 创建数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    valid_dataset = TensorDataset(
        torch.FloatTensor(X_valid),
        torch.LongTensor(y_valid)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    return train_loader, valid_loader, test_loader



def subsample_dataloader(loader, ratio=0.3, seed=2026):
    dataset = loader.dataset
    sample_size = max(1, int(len(dataset) * ratio))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=sample_size, replace=False)
    subset = Subset(dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
def train_model_RL(model, train_loader, val_loader, test_loader, model_name, logger, cur_mean, cur_std, dqn_agent, action_scale_map, H_matrix, rl_optimizer, batch_size, num_classes, write=False, common=True, n_epochs=50, new_domain=False, acc=0):
    model = model.to(device)
    rho_scale = 0.0001
    rho = rho_scale*acc
    old_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    rel_changes = {}
    shift_changes = {}
    logger.info(f"\n{'='*60}")
    logger.info(f"训练模型: {model_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    history = init_history_dict()
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 20
    current_state_vec = None
    activate_freq = 0.5
    for epoch in range(1, n_epochs + 1):
        train_loss, train_correct = 0.0, 0
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
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                break
            grad_flat = get_flat_grad(model).detach()
        else:
            grad_flat = current_state_vec[0]
        grad_input = grad_flat.unsqueeze(0)
        mean_input = cur_mean.unsqueeze(0)
        std_input = cur_std.unsqueeze(0)
        logger.info(f'grad_input shape:{grad_input.shape}, mean_input shape:{mean_input.shape}, std_input shape:{std_input.shape}')
        current_state_vec = (grad_input, mean_input, std_input)

        sorted_layers = sorted(H_matrix.items(), key=lambda item: item[1])
        total_layers = len(sorted_layers)

        action_idx = dqn_agent.select_action(current_state_vec, new_domain)
        current_state_vec = (grad_input.squeeze(0), mean_input.squeeze(0), std_input.squeeze(0))
        scale = action_scale_map[action_idx]
        if not new_domain:
            logger.info(f'activate freq:{activate_freq}')
            if activate_freq > big_threshold:
                logger.info('high activation frequency, LTP')
                scale *= 0.9
            elif activate_freq < small_threshold:
                logger.info('low activation frequency, LTD')
                scale /= 0.9
        logger.info(f'freeze scale:{scale}')
        freeze_count = int(total_layers * scale)
        layers_to_freeze = [name for name, _ in sorted_layers[:freeze_count]]
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        freeze_layers(model, layers_to_freeze)
        total_freq = 0
        batch_num = 0
        for batch_X, batch_y in train_loader:
            batch_num += 1
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs, activate_freq = model(batch_X)
            total_freq += activate_freq
            # print(f'outputs shape:{outputs.shape}') # outputs shape:torch.Size([80, 4])
            labels = F.one_hot(batch_y, num_classes)
            soft_labels =  (1 - rho) * labels + rho * outputs
            loss = criterion(outputs, soft_labels)
            # loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
        activate_freq = total_freq / batch_num
        next_state_vec = copy.deepcopy(current_state_vec)
        updates = calculate_layer_updates(model, prev_params)
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        record_layer_changes(history, model, prev_params, updates)
        for name, update_val in updates.items():
            if name not in layers_to_freeze:
                H_matrix[name] = 0.9 * H_matrix[name] + 0.1 * update_val # 平滑更新
        record_rl_state(
            history=history,
            epoch=epoch,
            H_matrix=H_matrix,
            layers_to_freeze=layers_to_freeze,
            action_idx=action_idx,
            scale=scale,
            freeze_count=freeze_count,
            total_layers=total_layers,
            activate_freq=activate_freq
        )
        if not new_domain:
            model.eval()
            for param in model.parameters():
                param.requires_grad = True
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)               
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                break   
            grad_flat = get_flat_grad(model).detach() # [total_params] 
            logger.info(f'grad_flat shape:{grad_flat.shape}') 
            next_state_vec = (grad_flat, cur_mean, cur_std)
            reward = train_acc
            dqn_agent.memory.append((current_state_vec, action_idx, reward, next_state_vec))
            logger.info(f'current_state_vec:{current_state_vec}, action_idx:{action_idx}, reward:{reward}, next_state_vec:{next_state_vec}')
                # 训练 DQN
            dqn_agent.train_step(batch_size=batch_size, optimizer=rl_optimizer)
            current_state_vec = copy.deepcopy(next_state_vec)
        if common and write:
            writer.add_scalar('Loss/train_batch', train_loss, epoch)
        model.eval()
        val_loss, val_correct = 0.0, 0
        if epoch == 6:
            for name, param in model.named_parameters():
                old = old_params[name]
                new = param.detach()
                
                # 计算新旧参数的 L2 范数
                old_norm = old.norm().item()
                new_norm = new.norm().item()
                
                # 计算参数更新量的 L2 范数
                delta = new - old
                delta_norm = delta.norm().item()
                
                # 计算相对变化率（添加小常数防止除零）
                rel_change = delta_norm / (old_norm + 1e-12)
                shift_changes[name] = rel_change
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch {epoch:3d}/{n_epochs}: LR: {current_lr:.6f} | '
                  f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        
        # 早停
        if patience_counter >= patience:
            logger.info(f"早停在 epoch {epoch}, 最佳验证准确率: {best_val_acc:.4f}")
            break
    for name, param in model.named_parameters():
        old = old_params[name]
        new = param.detach()
        
        # 计算新旧参数的 L2 范数
        old_norm = old.norm().item()
        new_norm = new.norm().item()
        
        # 计算参数更新量的 L2 范数
        delta = new - old
        delta_norm = delta.norm().item()
        
        # 计算相对变化率（添加小常数防止除零）
        rel_change = delta_norm / (old_norm + 1e-12)
        rel_changes[name] = rel_change
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试阶段
    test_acc, test_report, cm = evaluate_model(model, test_loader)
    history['test_acc'] = test_acc
    
    training_time = time.time() - start_time
    
    # 打印结果
    logger.info(f"\n{model_name} 结果:")
    logger.info(f"  测试准确率: {test_acc:.4f}")
    logger.info(f"  参数量: {count_parameters(model):,}")
    logger.info(f"  训练时间: {training_time:.2f}秒")
    logger.info(f"  最佳验证准确率: {best_val_acc:.4f}")
    
    # 详细分类报告
    logger.info("\n分类报告:")
    for cls_name in CLASS_NAMES:
        if cls_name in test_report:
            precision = test_report[cls_name]['precision']
            recall = test_report[cls_name]['recall']
            f1 = test_report[cls_name]['f1-score']
            logger.info(f"  {cls_name}: 精度={precision:.3f}, 召回率={recall:.3f}, F1={f1:.3f}")
    
    return test_acc, test_report, cm, history, count_parameters(model), training_time, rel_changes, shift_changes, model

def train_model(model, train_loader, val_loader, test_loader, model_name, logger, common=True, n_epochs=50, write=False):
    model = model.to(device)
    old_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    rel_changes = {}
    shift_changes = {}
    """训练和评估模型"""
    logger.info(f"\n{'='*60}")
    logger.info(f"训练模型: {model_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=10)
    
    history = init_history_dict()
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 20
    
    for epoch in range(1, n_epochs + 1):
        # 训练阶段
        model.train()
        train_loss, train_correct = 0.0, 0
        prev_params = {name: param.data.clone() for name, param in model.named_parameters()}
        if common and write:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    writer.add_scalar(f'grad_norm/{name}', grad_norm, epoch)
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        updates = calculate_layer_updates(model, prev_params)
        record_layer_changes(history, model, prev_params, updates)
        if common and write:
            writer.add_scalar('Loss/train_batch', train_loss, epoch)
        model.eval()
        val_loss, val_correct = 0.0, 0
        if epoch == 6:
            for name, param in model.named_parameters():
                old = old_params[name]
                new = param.detach()
                old_norm = old.norm().item()
                new_norm = new.norm().item()
                delta = new - old
                delta_norm = delta.norm().item()
                rel_change = delta_norm / (old_norm + 1e-12)
                shift_changes[name] = rel_change
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch {epoch:3d}/{n_epochs}: LR: {current_lr:.6f} | '
                  f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        
        # 早停
        if patience_counter >= patience:
            logger.info(f"早停在 epoch {epoch}, 最佳验证准确率: {best_val_acc:.4f}")
            break
    for name, param in model.named_parameters():
        old = old_params[name]
        new = param.detach()
        old_norm = old.norm().item()
        new_norm = new.norm().item()
        delta = new - old
        delta_norm = delta.norm().item()
        rel_change = delta_norm / (old_norm + 1e-12)
        rel_changes[name] = rel_change
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_acc, test_report, cm = evaluate_model(model, test_loader)
    history['test_acc'] = test_acc
    
    training_time = time.time() - start_time
    logger.info(f"\n{model_name} 结果:")
    logger.info(f"  测试准确率: {test_acc:.4f}")
    logger.info(f"  参数量: {count_parameters(model):,}")
    logger.info(f"  训练时间: {training_time:.2f}秒")
    logger.info(f"  最佳验证准确率: {best_val_acc:.4f}")
    logger.info("\n分类报告:")
    for cls_name in CLASS_NAMES:
        if cls_name in test_report:
            precision = test_report[cls_name]['precision']
            recall = test_report[cls_name]['recall']
            f1 = test_report[cls_name]['f1-score']
            logger.info(f"  {cls_name}: 精度={precision:.3f}, 召回率={recall:.3f}, F1={f1:.3f}")
    
    return test_acc, test_report, cm, history, count_parameters(model), training_time, rel_changes, shift_changes, model

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs, _ = model(batch_X)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    class_names = CLASS_NAMES
    labels = list(range(len(class_names)))

    report = classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # labels=[0,1] ensures cm = [[TN, FP], [FN, TP]]
    cm = confusion_matrix(all_labels, all_preds, labels=labels)

    return accuracy, report, cm
backbone_name = 'EEGNet'
args, logger, now = get_config(backbone_name)
def main():
    n_clusters = 2
    train_loaders = []
    val_loaders = []
    test_loaders = []
    model_name = 'BrainAI'
    OUTPUT_DIR = f"./EEG_output/{model_name}/{backbone_name}_results_classification"
    BATCH_SIZE = 80
    N_EPOCHS = 800
    N_SUBJECTS = 7
    new_i = 6
    add_idx = []
    target_subject_name = f"subject_{new_i + 1:02d}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    run_seed = 2026
    num_experiments = 10
    args, logger, now = get_config(backbone_name, dataset_for_log=dataset)
    result_dir = make_result_dir(dataset, backbone_name, now)
    logger.info(f'Result directory: {result_dir}')
    logger.info(f'使用的backbone: {backbone_name}')
    logger.info(f'Dataset: {dataset}')
    logger.info(f'N_SUBJECTS={N_SUBJECTS}, target/new domain={target_subject_name}, add_idx={add_idx}')

    cur_means = []
    cur_stds = []
    for subject_id in range(N_SUBJECTS):
        DATA_PATH = f"./data/data/chbmit_processed/subject_{subject_id+1:02d}_preprocessed.npz"
        logger.info(f"\n1. 加载 CHB-MIT 预处理数据 subject_{subject_id+1:02d}...")
        try:
            X_valid, y_valid, X_train, y_train, X_test, y_test, cur_mean, cur_std = load_preprocessed_data(DATA_PATH)
        except Exception as e:
            logger.info(f"加载数据失败: {e}")
            return

        logger.info("\n2. 创建数据加载器...")
        try:
            train_loader, val_loader, test_loader = create_dataloaders(
                X_train, y_train, X_valid, y_valid, X_test, y_test,
                batch_size=BATCH_SIZE
            )
        except Exception as e:
            logger.info(f"创建数据加载器失败: {e}")
            return

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
        cur_means.append(cur_mean)
        cur_stds.append(cur_std)

        n_channels = X_train.shape[1]
        n_classes = len(CLASS_NAMES)
        input_time_length = X_train.shape[2]
        logger.info(
            f"subject_{subject_id+1:02d}: "
            f"n_channels={n_channels}, n_classes={n_classes}, "
            f"input_time_length={input_time_length}, "
            f"train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}"
        )
    models_to_test = {
        "EEGNet": EEGNetPaper(n_channels, n_classes, input_time_length),
        "TCN": FixedTCN(n_channels, n_classes, input_time_length),
        "CompactCNN": CompactCNNPaper(n_channels, n_classes, input_time_length),
        "MLP": MLPModel(n_channels, n_classes, input_time_length),
        "Deep4Net": Deep4NetWrapper(n_channels, n_classes, input_time_length),
        "ShallowNet": ShallowFBCSPNetWrapper(n_channels, n_classes, input_time_length)
    }
    del X_train, y_train, X_valid, y_valid, X_test, y_test

    results = {}
    all_test_acc = []
    for experiment_idx in range(num_experiments):
        logger.info(f'======================第{experiment_idx+1}/{num_experiments}次实验开始======================')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                f"可用GPU内存: "
                f"{torch.cuda.memory_allocated()/1024**3:.1f} GB / "
                f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB"
            )

        common_layer_changes = [0 for _ in range(len(train_loaders))]
        common_shift_changes = [0 for _ in range(len(train_loaders))]
        sum_gradients = []
        cat_gradients = {}
        source_subject_results = {}
        for i in range(len(train_loaders)):
            model = models_to_test[backbone_name]
            train_loader = train_loaders[i]
            val_loader = val_loaders[i]
            test_loader = test_loaders[i]

            if i == new_i:
                continue

            logger.info(f'start training source subject A{i+1:02d}')
            test_acc, test_report, cm, history, params, training_time, curriculum_rel_changes, curriculum_shift_changes, model = train_model(
                model, train_loader, val_loader, test_loader,
                model_name, logger, common=False, n_epochs=N_EPOCHS // 10
            )

            sum_gradient, cat_gradient = compute_gradient_EEG(model, val_loader, device, criterion)
            logger.info(f"sum gradient: {sum_gradient}, cat_grad_len: {len(cat_gradient)}")
            sum_gradients.append(sum_gradient)
            cat_gradients[i] = cat_gradient

            source_subject_results[f'A{i+1:02d}'] = {
                'test_acc_on_own_E_session': float(test_acc),
                'params': int(params),
                'training_time': float(training_time),
                'sum_gradient': float(sum_gradient)
            }

            results[model_name] = {
                'test_acc': test_acc,
                'test_report': test_report,
                'confusion_matrix': cm,
                'history': history,
                'params': params,
                'time': training_time
            }
        cluster_labels, cluster_members = clustering_dict(n_clusters=n_clusters, data_dict=cat_gradients, algorithm='kmeans')
        logger.info(f'cluster members: {cluster_members}')

        cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_gradients)
        logger.info(f'cluster grad: {cluster_grad}')

        sum_gradients_cluster = [sum(cluster_grad[i]) for i in range(len(cluster_grad))]
        sorted_clusters_gradients_list = sort_by_original_gradient(sum_gradients=sum_gradients_cluster)
        subject_to_cluster = build_subject_to_cluster(cluster_members)

        p0 = 0.1
        common_model = models_to_test[backbone_name]
        num_circles = 3
        sorted_sum_gradients = []
        sorted_cat_gradients = []
        first_step_trace = []

        for circle_index in range(1):
            ordered_subjects = []
            for cluster_index in sorted_clusters_gradients_list:
                cluster = cluster_members[cluster_index]
                temp_sorted_gradients_list = sort_by_original_gradient(
                    sum_gradients=[sum_gradients[domain] for domain in cluster]
                )
                sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
                logger.info(f'sorted_gradients_list: {sorted_gradients_list}')
                ordered_subjects.extend(sorted_gradients_list)

            for i in ordered_subjects:
                grad_norms = {name: [] for name, param in common_model.named_parameters()}
                sorted_sum_gradients.append(sum_gradients[i])
                sorted_cat_gradients.append(cat_gradients[i])
                cluster_index = subject_to_cluster.get(int(i), -1)
                cluster = cluster_members[cluster_index] if cluster_index in cluster_members else []
                if len(cluster) > 0:
                    d_max = max(sum_gradients[domain] for domain in cluster)
                else:
                    d_max = max(sum_gradients)

                if (i == new_i) or (i in add_idx):
                    first_step_trace.append({
                        'circle_index': int(circle_index),
                        'cluster_index': int(cluster_index),
                        'subject': f'subject_{i+1:02d}',
                        'skipped': True,
                        'reason': 'target_or_add_idx'
                    })
                    continue

                logger.info(f'subject A{i+1:02d} start training...')
                train_loader = train_loaders[i]
                val_loader = val_loaders[i]
                test_loader = test_loaders[i]
                cur_mean = torch.from_numpy(cur_means[i]).float().to(device)
                cur_std = torch.from_numpy(cur_stds[i]).float().to(device)
                difference_value = sum_gradients[i]

                if cluster_index >= 0:
                    p_common = my_sigmoid(p0, sum_gradients_cluster[cluster_index], d_max)
                else:
                    p_common = p0
                common_model.dropout = p_common
                first_step_trace.append({
                    'circle_index': int(circle_index),
                    'cluster_index': int(cluster_index),
                    'subject': f'subject_{i+1:02d}',
                    'skipped': False,
                    'dropout_p': float(p_common),
                    'subject_sum_gradient': float(sum_gradients[i]),
                    'cluster_sum_gradient': float(sum_gradients_cluster[cluster_index]) if cluster_index >= 0 else None,
                    'd_max': float(d_max),
                })

                test_acc, test_report, cm, history, params, training_time, curriculum_rel_changes, curriculum_shift_changes, common_model = train_model(
                    common_model, train_loader, val_loader, test_loader,
                    model_name, logger, common=False, n_epochs=N_EPOCHS
                )
                logger.info(f'common_layer_changes on subject_A{i+1:02d}: {common_layer_changes[i]}')
                logger.info(f'common_shift_changes on subject_A{i+1:02d}: {common_shift_changes[i]}')
        num_circles = 3
        sorted_sum_gradients = []
        sorted_cat_gradients = []
        common_stage_num_epochs = 50
        common_learning_rate = 0.001
        weight_decay_common = 0.01
        total_params = sum(param.nelement() for param in common_model.parameters())
        action_scale_map = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        state_dim = 16
        dqn_agent = DQNAgent(
            grad_dim=total_params,
            encoding_dim=n_channels,
            data_dim=n_channels,
            state_dim=state_dim,
            action_dim=len(action_scale_map),
            device=device,
            logger=logger
        ).to(device)
        rl_optimizer = optim.Adam(dqn_agent.q_net.parameters(), lr=0.001)
        H_matrix = {name: 0.0 for name, p in common_model.named_parameters()}
        test_acc = 0
        common_model.dropout = p0

        second_step_trace = []
        learning_loop_trace = []
        for circle_index in range(num_circles):
            ordered_subjects = []
            for cluster_index in sorted_clusters_gradients_list:
                cluster = cluster_members[cluster_index]
                temp_sorted_gradients_list = sort_by_original_gradient(
                    sum_gradients=[sum_gradients[domain] for domain in cluster]
                )
                sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
                logger.info(f'sorted_gradients_list: {sorted_gradients_list}')
                ordered_subjects.extend(sorted_gradients_list)

            for i in ordered_subjects:
                grad_norms = {name: [] for name, param in common_model.named_parameters()}
                sorted_sum_gradients.append(sum_gradients[i])
                sorted_cat_gradients.append(cat_gradients[i])

                if i == new_i:
                    continue

                cluster_index = subject_to_cluster.get(int(i), -1)
                logger.info(f'subject A{i+1:02d} start training...')
                train_loader = train_loaders[i]
                val_loader = val_loaders[i]
                test_loader = test_loaders[i]
                cur_mean = torch.from_numpy(cur_means[i]).float().to(device)
                cur_std = torch.from_numpy(cur_stds[i]).float().to(device)
                difference_value = sum_gradients[i]

                test_acc, test_report, cm, history, params, training_time, common_layer_changes[i], common_shift_changes[i], common_model = train_model_RL(
                    common_model, train_loader, val_loader, test_loader,
                    model_name, logger, cur_mean, cur_std, dqn_agent, action_scale_map,
                    H_matrix, rl_optimizer, batch_size=BATCH_SIZE, num_classes=n_classes,
                    common=True, n_epochs=N_EPOCHS, acc=test_acc
                )
                second_step_trace.append({
                    'circle_index': int(circle_index),
                    'cluster_index': int(cluster_index),
                    'subject_index': int(i),
                    'subject': f'subject_{i+1:02d}',
                    'test_acc_on_own_E_session': float(test_acc),
                    'training_time': float(training_time),
                    'num_rl_epochs_recorded': int(len(history.get('rl_action_history', []))),
                })

                source_loop_entries = build_learning_loop_entries(
                    history,
                    metadata={
                        'experiment_idx': int(experiment_idx + 1),
                        'phase': 'source_rl_train',
                        'circle_index': int(circle_index),
                        'cluster_index': int(cluster_index),
                        'subject_index': int(i),
                        'subject': f'subject_{i+1:02d}',
                        'target_subject': target_subject_name,
                        'new_domain': False,
                        }
                )
                learning_loop_trace.extend(source_loop_entries)

                logger.info(f'common_layer_changes on subject_A{i+1:02d}: {common_layer_changes[i]}')
                logger.info(f'common_shift_changes on subject_A{i+1:02d}: {common_shift_changes[i]}')
        train_loader = train_loaders[new_i]
        original_target_train_size = len(train_loader.dataset)
        train_loader = subsample_dataloader(
            train_loader,
            ratio=0.3,
            seed=run_seed + experiment_idx
        )
        logger.info(
            f'Target domain train loader downsampled to 30%: '
            f'{len(train_loader.dataset)}/{original_target_train_size} samples'
        )
        val_loader = val_loaders[new_i]
        test_loader = test_loaders[new_i]
        cur_mean = torch.from_numpy(cur_means[new_i]).float().to(device)
        cur_std = torch.from_numpy(cur_stds[new_i]).float().to(device)

        test_acc, test_report, cm, history, params, training_time, new_layer_changes, new_shift_changes, common_model = train_model_RL(
            common_model, train_loader, val_loader, test_loader,
            model_name, logger, cur_mean, cur_std, dqn_agent, action_scale_map,
            H_matrix, rl_optimizer, batch_size=BATCH_SIZE, common=True,
            num_classes=n_classes, n_epochs=N_EPOCHS, new_domain=True
        )
        target_learning_loop_trace = build_learning_loop_entries(
            history,
            metadata={
                'experiment_idx': int(experiment_idx + 1),
                'phase': 'target_new_domain_adaptation',
                'circle_index': None,
                'cluster_index': None,
                'subject_index': int(new_i),
                'subject': target_subject_name,
                'target_subject': target_subject_name,
                'new_domain': True,
            }
        )

        all_test_acc.append(test_acc)

        exp_summary = save_experiment_results(
            result_dir=result_dir,
            experiment_idx=experiment_idx + 1,
            dataset_name=dataset,
            backbone_name=backbone_name,
            target_subject=new_i + 1,
            model_name=model_name,
            test_acc=test_acc,
            test_report=test_report,
            cm=cm,
            history=history,
            params=params,
            training_time=training_time,
            logger=logger
        )
        exp_dir = os.path.join(result_dir, f"exp_{experiment_idx + 1}")
        torch.save(common_model.state_dict(), os.path.join(exp_dir, 'final_common_model_state_dict.pth'))
        save_learning_loop_trace(
            exp_dir=exp_dir,
            learning_loop_trace=learning_loop_trace,
            target_learning_loop_trace=target_learning_loop_trace,
            logger=logger
        )

        experiment_trace = {
            'experiment_idx': int(experiment_idx + 1),
            'dataset': dataset,
            'backbone': backbone_name,
            'target_subject': target_subject_name,
            'source_subjects': [f'subject_{i+1:02d}' for i in range(N_SUBJECTS) if i != new_i],
            'new_i': int(new_i),
            'add_idx': [int(x) for x in add_idx],
            'n_clusters': int(n_clusters),
            'source_subject_results': source_subject_results,
            'cluster_labels': to_builtin(cluster_labels),
            'cluster_members': to_builtin(cluster_members),
            'sum_gradients': [float(x) for x in sum_gradients],
            'sum_gradients_cluster': [float(x) for x in sum_gradients_cluster],
            'sorted_clusters_gradients_list': to_builtin(sorted_clusters_gradients_list),
            'first_step_trace': first_step_trace,
            'second_step_trace': second_step_trace,
            'source_learning_loop_trace_file': 'source_learning_loop_trace.json',
            'source_learning_loop_trace_csv': 'source_learning_loop_trace.csv',
            'target_learning_loop_trace_file': 'target_learning_loop_trace.json',
            'num_source_learning_loop_steps': int(len(learning_loop_trace)),
            'num_target_learning_loop_steps': int(len(target_learning_loop_trace)),
            'target_result_summary': exp_summary,
            'new_layer_changes': to_builtin(new_layer_changes),
            'new_shift_changes': to_builtin(new_shift_changes)
        }
        save_json(experiment_trace, os.path.join(exp_dir, 'experiment_trace.json'))

        results[model_name] = {
            'test_acc': test_acc,
            'test_report': test_report,
            'confusion_matrix': cm,
            'history': history,
            'params': params,
            'time': training_time
        }
        if not results:
            logger.info("没有模型成功训练")
            return
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
        for model_name, result in sorted_results:
            logger.info(f"{model_name} on {target_subject_name}:")
            logger.info(f"  测试准确率: {result['test_acc']:.4f}")
            logger.info(f"  参数量: {result['params']:,}")
            logger.info(f"  训练时间: {result['time']:.2f}秒")
    mean_acc = sum(all_test_acc) / len(all_test_acc)
    logger.info(f"平均值: {round(mean_acc, 4)}")
    save_overall_summary(
        result_dir=result_dir,
        dataset_name=dataset,
        backbone_name=backbone_name,
        all_test_acc=all_test_acc,
        logger=logger
    )
if __name__ == "__main__":
    try:
        main()
    except:
        logger.exception('Exception')
