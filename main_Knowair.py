

import os, sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

from util import config, file_dir
from graph import Graph
from dataset_per_channel_month import few_shot_sample, get_channel_month_2d_loaders
from utils.logging import *
from utils.utils import *
from utils.metrics import *
from tools.gradient_compute import *
from tools.cluster_ST import *
from model.DQN import *
from model_air.MLP import MLP
from model_air.LSTM import LSTM
from model_air.GRU import GRU
from model_air.GC_LSTM import GC_LSTM
from model_air.nodesFC_GRU import nodesFC_GRU
from model_air.PM25_GNN import PM25_GNN
from model_air.PM25_GNN_nosub import PM25_GNN_nosub
from utils.args import *
import arrow, torch
from torch import nn
import numpy as np
import datetime
import pickle
import json
import csv
from torch.utils.tensorboard import SummaryWriter
from visualize_kepler_fixed import save_station_coords, visualize_predictions_kepler, visualize_domain_comparison

dataset = 'Knowair'
writer_dir = f'Writer/BrainAI/{dataset}'
big_threshold = 0.8
small_threshold = 0.2
if not os.path.exists(writer_dir):
    os.makedirs(writer_dir)
writer = SummaryWriter(writer_dir) 
torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
graph        = Graph()
city_num     = graph.node_num
batch_size   = config['train']['batch_size']
epochs       = config['train']['epochs']
hist_len     = config['train']['hist_len']
pred_len     = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop   = config['train']['early_stop']
lr           = config['train']['lr']
results_dir  = file_dir['results_dir']
dataset_num  = config['experiments']['dataset_num']
exp_model    = config['experiments']['model']
exp_repeat   = config['train']['exp_repeat']
save_npy     = config['experiments']['save_npy']
metero_use   = config['experiments']['metero_use']
# STATION_COORDS_FILE = os.path.join(proj_dir, 'data', 'knowair_station_coords.json')
STATION_COORDS_FILE = os.path.join(proj_dir, 'data', 'station_info.csv')
VISUALIZATION_OUTPUT_DIR = os.path.join(proj_dir, 'visualization_results')
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
        'train_loss': [],
        'val_loss': [], 
        'test_mae': 0.0,
        # For layer-change heatmaps
        'layer_update_norms': {},
        'layer_relative_changes': {},
        # For hippocampus / RL visualization
        'H_matrix_history': [],
        'frozen_layers_history': [],
        'rl_action_history': [],
        'activate_freq_history': [],
        'visualization_saved': False,
        'domain_info': {}
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


def save_experiment_results(result_dir, experiment_idx, dataset_name, backbone_name,
                            target_subject, model_name, test_mae, history, logger=None):
    """Save one experiment for later paper tables and figures."""
    target_subject_i = target_subject[0]
    target_subject_j = target_subject[1]
    exp_dir = os.path.join(result_dir, f"exp_{experiment_idx}")
    ensure_dir(exp_dir)

    save_json(history, os.path.join(exp_dir, "history.json"))
    save_pickle(history, os.path.join(exp_dir, "history.pkl"))

    train_loss = history.get('train_loss', []) if isinstance(history, dict) else []
    val_loss = history.get('val_loss', []) if isinstance(history, dict) else []

    exp_summary = {
        'experiment_idx': int(experiment_idx),
        'dataset': str(dataset_name),
        'backbone': str(backbone_name),
        'model_name': str(model_name),
        'target_subject_i': int(target_subject_i),
        'target_subject_j': int(target_subject_j),
        'test_mae': float(test_mae),
        'final_train_loss': float(train_loss[-1]) if len(train_loss) > 0 else 0.0,
        'final_val_loss': float(val_loss[-1]) if len(val_loss) > 0 else 0.0,
        'visualization_saved': bool(history.get('visualization_saved', False)),
    }

    save_json(exp_summary, os.path.join(exp_dir, "experiment_summary.json"))
    append_csv_row(
        os.path.join(result_dir, "summary.csv"),
        fieldnames=[
            'experiment_idx', 'dataset', 'backbone', 'model_name', 'target_subject_i','target_subject_j', 'test_mae', 'final_train_loss', 'final_val_loss','visualization_saved'
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

def get_config():
    parser = create_parser()
    args = parser.parse_args()
    
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    log_dir = f'logs/BrainAI/Knowair/{exp_model}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(log_dir, __name__, '{}.log'.format(now))
    logger.info(args)
    
    return args, logger, now
args, logger, now = get_config()
(train_loaders, val_loaders, test_loaders,
            cur_means, cur_stds,
            pm25_mean, pm25_std,
            wind_mean, wind_std,
            domain_names, in_dim_per_domain) = get_channel_month_2d_loaders(
    graph, hist_len, pred_len, dataset_num, batch_size, logger=logger
)

channel_num = len(train_loaders)

if not os.path.exists(STATION_COORDS_FILE):
    logger.info("First run: extracting and saving station coordinates...")
    save_station_coords(graph, STATION_COORDS_FILE)
    logger.info(f"Station coordinates saved to: {STATION_COORDS_FILE}")
else:
    logger.info(f"Station coordinate file already exists: {STATION_COORDS_FILE}")

sample_pm25, sample_feat, _ = next(iter(train_loaders[0][0]))
in_dim = 6 # 1+4+1
logger.info(f"in_dim={in_dim}  (feat_dim={sample_feat.shape[-1]}, "
      f"pm25_dim={sample_pm25.shape[-1]})")

criterion = nn.MSELoss()
def get_metric(predict_epoch, label_epoch):
    """
    predict_epoch / label_epoch : (N, seq_len, 1, 1)
    MAPE is computed as a weighted percentage variant:
    sum(abs(pred - label)) / sum(abs(label)) * 100.
    """
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss) if (hit + falsealarm + miss) else 0
    pod = hit / (hit + miss) if (hit + miss) else 0
    far = falsealarm / (hit + falsealarm) if (hit + falsealarm) else 0

    predict = predict_epoch[:, :, :, 0].transpose((0, 2, 1)).reshape(-1, predict_epoch.shape[1])
    label = label_epoch[:, :, :, 0].transpose((0, 2, 1)).reshape(-1, label_epoch.shape[1])
    abs_error = np.abs(predict - label)
    mae = np.mean(np.mean(abs_error, axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    label_sum = np.sum(np.abs(label))
    mape = np.sum(abs_error) / label_sum * 100 if label_sum > 0 else np.nan
    return rmse, mae, mape, csi, pod, far

def _patch_pm25_gnn(model):
    _orig_edge_fwd = model.graph_gnn.edge_mlp.forward
    def _safe_edge_fwd(x):
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return _orig_edge_fwd(x_clean)
    model.graph_gnn.edge_mlp.forward = _safe_edge_fwd

    _orig_gnn_fwd = model.graph_gnn.forward
    def _safe_gnn_fwd(*args, **kwargs):
        out = _orig_gnn_fwd(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out
    model.graph_gnn.forward = _safe_gnn_fwd

    return model
def get_model():
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'PM25_GNN':
        return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_nosub':
        return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    else:
        raise Exception('Wrong model name!')
def test_model(test_loader, model):
    best_p, best_l, best_t = test_one_city(test_loader, model)
    rmse, mae, mape, csi, pod, far = get_metric(best_p, best_l)
    return rmse, mae, mape, csi, pod, far
def train_one_epoch(loader, model, optimizer):
    model.train()
    total = 0
    for pm25, feature, _ in loader:
        optimizer.zero_grad()
        pm25    = pm25.to(device)  
        feature = feature.to(device) 

        pred, _ = model(pm25[:, :hist_len], feature)
        loss = criterion(pred, pm25[:, hist_len:])
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)
def train_model_RL(model, train_loader, val_loader, test_loader, model_name, logger, cur_mean, cur_std, dqn_agent, action_scale_map, H_matrix, 
                   rl_optimizer, batch_size, i, j, write=False, 
                   common=True, n_epochs=50, new_domain=False, acc=0):
    model = model.to(device)
    history = init_history_dict()
    history['domain_info'] = {
        'channel': int(i),
        'month': int(j),
        'is_new_domain': bool(new_domain),
        'metero_name': str(metero_use[i]) if i < len(metero_use) else f'meteo_{i}'
    }
    rho = 0.0001
    logger.info(f'rho value: {rho}')
    old_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    rel_changes = {}
    shift_changes = {}
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float('inf')
    cnt = 0
    best_p, best_l, best_t = None, None, None
    criterion = nn.MSELoss()
    current_state_vec = None
    activate_freq = 0.5
    total_loss = 0
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
            for pm25, feature, _ in train_loader:
                optimizer.zero_grad()
                pm25    = pm25.to(device)    
                feature = feature.to(device) 

                pred,_ = model(pm25[:, :hist_len], feature)
                loss = criterion(pred, pm25[:, hist_len:])
                loss.backward()
                break
            grad_flat = get_flat_grad(model).detach()
        else:
            grad_flat = current_state_vec[0]
        grad_input = grad_flat.unsqueeze(0) # [1, total_params]
        mean_input = cur_mean.unsqueeze(0)  # [1, 11]
        std_input = cur_std.unsqueeze(0)
        logger.info(f'grad_input shape:{grad_input.shape}, mean_input shape:{mean_input.shape}, std_input shape:{std_input.shape}')
        current_state_vec = (grad_input, mean_input, std_input) 
        action_idx = dqn_agent.select_action(current_state_vec, new_domain)
        current_state_vec = (grad_input.squeeze(0), mean_input.squeeze(0), std_input.squeeze(0)) 
        scale = action_scale_map[action_idx]
        if not new_domain:
            logger.info(f'activate freq:{activate_freq}')
            if activate_freq > big_threshold:
                logger.info('High activation frequency, LTP')
                scale *= 0.9
            elif activate_freq < small_threshold:
                logger.info('Low activation frequency, LTD')
                scale /= 0.9
        logger.info(f'freeze scale:{scale}')
        sorted_layers = sorted(H_matrix.items(), key=lambda item: item[1])
        total_layers = len(sorted_layers)
        freeze_count = int(total_layers * scale)
        layers_to_freeze = [name for name, _ in sorted_layers[:freeze_count]]
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if name in layers_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        total_freq = 0
        batch_num = 0
        for pm25, feature, _ in train_loader:  # normal training
            optimizer.zero_grad()
            pm25    = pm25.to(device)    # (B, seq_len, 1) -> (B, seq_len, 1, 1)
            feature = feature.to(device) # (B, seq_len, D) -> (B, seq_len, 1, D)
            batch_num += 1
            pred, activate_freq = model(pm25[:, :hist_len], feature)
            total_freq += activate_freq
            label = pm25[:, hist_len:]
            target = (1 - rho) * label + rho * pred
            # loss = criterion(pred, pm25[:, hist_len:])
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        activate_freq = total_freq / batch_num
        next_state_vec = copy.deepcopy(current_state_vec)
        updates = calculate_layer_updates(model, prev_params)
        record_layer_changes(history, model, prev_params, updates)
        for name, update_val in updates.items():
            if name not in layers_to_freeze:
                H_matrix[name] = 0.9 * H_matrix[name] + 0.1 * update_val # smooth update
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
            for pm25, feature, _ in train_loader:
                optimizer.zero_grad()
                pm25    = pm25.to(device)    # (B, seq_len, 1) -> (B, seq_len, 1, 1)
                feature = feature.to(device) # (B, seq_len, D) -> (B, seq_len, 1, D)

                pred, _ = model(pm25[:, :hist_len], feature)
                loss = criterion(pred, pm25[:, hist_len:])
                loss.backward()
                break
            grad_flat = get_flat_grad(model).detach() 
            next_state_vec = (grad_flat, cur_mean, cur_std)
            reward = 1 / total_loss
            dqn_agent.memory.append((current_state_vec, action_idx, reward, next_state_vec))

            dqn_agent.train_step(batch_size=batch_size, optimizer=rl_optimizer)
            current_state_vec = copy.deepcopy(next_state_vec)
        va = val_one_city(val_loader, model)
        history['train_loss'].append(total_loss)
        history['val_loss'].append(va)
        if va < best_val:
            best_val = va
            cnt = 0
            best_p, best_l, best_t = test_one_city(test_loader, model)
        else:
            cnt += 1

        logger.info(f"  metero {i:3d}, month {j}"
                f"Ep {epoch+1:3d}/{epochs}  "
                f"val={va:.4f}  "
                f"best_val={best_val:.4f}  "
                f"patience={cnt}/{early_stop}")

        if cnt >= early_stop:
            logger.info(f"  metero {i:3d}, month {j}"
                    f"-> Early stop at epoch {epoch+1}")
            break
        
    rmse, mae, mape, csi, pod, far = get_metric(best_p, best_l)
    history['mae'] = mae
    logger.info(f"\n  >>> metero {i:3d}, month {j}"
            f"RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.4f} "
            f"CSI={csi:.4f}  POD={pod:.4f}  FAR={far:.4f}")
    if save_npy:
        d = os.path.join(
            results_dir,
            f'{exp_model}_dataset{dataset_num}_metero{i}_month{j}_{now}'
        )
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, 'predict.npy'), best_p)
        np.save(os.path.join(d, 'label.npy'),   best_l)
        np.save(os.path.join(d, 'time.npy'),    best_t)
        logger.info(f"      Saved -> {d}")

        vis_dir = os.path.join(d, 'kepler_vis')
        os.makedirs(vis_dir, exist_ok=True)

        norm_params = {
            'pm25_mean': float(pm25_mean),
            'pm25_std': float(pm25_std),
            'hist_len': int(hist_len),
            'pred_len': int(pred_len),
            'domain_channel': int(i),
            'domain_month': int(j),
            'domain_name': f'meteo{i}_month{j}'
        }
        save_json(norm_params, os.path.join(vis_dir, 'norm_params.json'))
        logger.info(f"Visualization parameters saved -> {vis_dir}")
    return rmse, mae, mape, csi, pod, far, model, history
def train_one_city(loader, model, optimizer, epochs):

    model.train()
    for epoch in range(epochs):
        loss = train_one_epoch(loader, model, optimizer)
        logger.info(f'Epoch: {epoch}, Loss: {loss}')

def val_one_city(loader, model):
    model.eval()
    total = 0
    with torch.no_grad():
        for pm25, feature, _ in loader:
            pm25    = pm25.to(device)
            feature = feature.to(device)

            pred, _ = model(pm25[:, :hist_len], feature)
            total += criterion(pred, pm25[:, hist_len:]).item()
    return total / len(loader)

def test_one_city(loader, model):
    model.eval()
    preds, labels, times = [], [], []
    with torch.no_grad():
        for pm25, feature, t in loader:
            pm25    = pm25.to(device)    # (B, seq_len, 1, 1)
            feature = feature.to(device) # (B, seq_len, 1, D)

            hist = pm25[:, :hist_len]                 # (B, hist_len, 1, 1)
            pred, _ = model(hist, feature)                # (B, pred_len, 1, 1)

            # denormalize
            pred_val = np.concatenate(
                [hist.cpu().numpy(), pred.cpu().numpy()], axis=1
            ) * pm25_std + pm25_mean                   # (B, seq_len, 1, 1)
            label_val = pm25.cpu().numpy() * pm25_std + pm25_mean  # (B, seq_len, 1, 1)

            preds.append(pred_val)
            labels.append(label_val)
            times.append(t.numpy())
    p = np.concatenate(preds)
    p[p < 0] = 0
    return p, np.concatenate(labels), np.concatenate(times)


def main():
    new_i = channel_num - 1
    new_j = 11
    cross_channel = True
    cross_temporal = False
    assert cross_channel != cross_temporal
    few_shot_scale = 0.3
    logger.info(f'few shot scale:{few_shot_scale}')
    logger.info(f'new domain index: {new_i}')
    exp_time = arrow.now().format('YYYYMMDDHHmmss')
    logger.info(f"\n{'='*60}")
    logger.info(f"  Model: {exp_model}  |  city_num: {city_num}  |  in_dim: {in_dim}")
    logger.info(f"  batch_size: {batch_size}  |  epochs: {epochs}  |  lr: {lr}")
    logger.info(f"{'='*60}\n")
    result_dir = make_result_dir(dataset, exp_model, now)
    logger.info(f'Result directory: {result_dir}')
    logger.info(f'Backbone: {exp_model}')
    n_clusters = 2
    for i in range(channel_num):
        for j in range(11):
            if i == new_i and j == new_j:
                train_loaders[i][j] = few_shot_sample(train_loaders[i][j], few_shot_scale)
            if i != new_i:
                logger.info(f"  metero {i:3d}, month {j}: {metero_use[i]:10s}  "
                    f"train={len(train_loaders[i][j].dataset):5d}  "
                    f"val={len(val_loaders[i][j].dataset):5d}  "
                    f"test={len(test_loaders[i][j].dataset):5d}")
    all_test_mae = []
    all_test_mape = []
    all_test_rmse = []
    all_domain_results = []
    for exp_idx in range(exp_repeat):
        logger.info(f"=================== Experiment {exp_idx+1}/{exp_repeat} results ===================")
        sum_gradients = []
        cat_gradients = []
        for i in range(channel_num):
            sum_gradients.append([])
            cat_gradients.append([])
            for j in range(11):
                if cross_channel: 
                    if i == new_i : continue
                if cross_temporal:
                    if j == new_j : continue
                model = get_model().to(device)
                optimizer = torch.optim.Adam(model.parameters(),
                                            lr=lr, weight_decay=weight_decay)
                best_val = float('inf')
                cnt = 0
                best_p, best_l, best_t = None, None, None
                train_loader = train_loaders[i][j]
                val_loader = val_loaders[i][j]
                test_loader = test_loaders[i][j]
                for epoch in range(1):
                    tr = train_one_epoch(train_loader, model, optimizer)
                    va = val_one_city(val_loader, model)

                    if va < best_val:
                        best_val = va
                        cnt = 0
                        best_p, best_l, best_t = test_one_city(test_loader, model)
                    else:
                        cnt += 1

                    logger.info(f"  metero {i:3d}, month {j}: {metero_use[i]:10s}  "
                        f"Ep {epoch+1:3d}/{epochs}  "
                        f"train={tr:.4f}  val={va:.4f}  "
                        f"best_val={best_val:.4f}  "
                        f"patience={cnt}/{early_stop}")

                    if cnt >= early_stop:
                        logger.info(f"  metero {i:3d}, month {j}: {metero_use[i]:10s}  "
                            f"-> Early stop at epoch {epoch+1}")
                        break
                    
                rmse, mae, mape, csi, pod, far = get_metric(best_p, best_l)
                logger.info(f"\n  >>> metero {i:3d}, month {j}: {metero_use[i]:10s}  "
            f"RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.4f} "
                    f"CSI={csi:.4f}  POD={pod:.4f}  FAR={far:.4f}")
                sum_gradient, cat_gradient = compute_gradient_air(model, val_loader, device, criterion, hist_len) 
                logger.info(f"sum gradient: {sum_gradient}, cat_grad_len: {len(cat_gradient)}")
                sum_gradients[i].append(sum_gradient)
                cat_gradients[i].append(cat_gradient) 
                if save_npy:
                    d = os.path.join(
                        results_dir,
                        f'{exp_model}_dataset{dataset_num}_metero{i}_{metero_use[i]}_month{j}_{exp_time}'
                    )
                    os.makedirs(d, exist_ok=True)
                    np.save(os.path.join(d, 'predict.npy'), best_p)
                    np.save(os.path.join(d, 'label.npy'),   best_l)
                    np.save(os.path.join(d, 'time.npy'),    best_t)
                    logger.info(f"      Saved -> {d}")
                    all_domain_results.append({
                        'domain': (i, j),
                        'result_dir': d
                    })
        cluster_members = clustering_2d_list(n_clusters=n_clusters, tensor_2d_list=cat_gradients, algorithm='kmeans')
        logger.info(f'cluster members: {cluster_members}')
        cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_gradients)
        logger.info(f'cluster grad: {cluster_grad}')
        # min_grad, min_i,min_j = get_min_gradient(sum_gradients=cluster_grad, new_i=-1, new_j=-1)
        sum_gradients_cluster = [sum(cluster_grad[i]) for i in range(len(cluster_grad))]
        sorted_clusters_gradients_list = sort_by_original_gradient(sum_gradients=sum_gradients_cluster)
        p0 = 0.1
        common_model = get_model().to(device)
        num_circles = 1
        sorted_sum_gradients = []
        sorted_cat_gradients = []
        for circle_index in range(1):
            for cluster_index in sorted_clusters_gradients_list:
                cluster = cluster_members[cluster_index]  # domains in this cluster
                temp_gradients_list = []
                for i in range(len(cluster)):
                    temp_gradients_list.append(sum_gradients[cluster[i][0]][cluster[i][1]])
                temp_sorted_gradients_list = sort_by_original_gradient(sum_gradients=temp_gradients_list)
                sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
                logger.info(f'sorted_gradients_list: {sorted_gradients_list}')
                d_max = sum_gradients[sorted_gradients_list[-1][0]][sorted_gradients_list[-1][1]]
                for i_j_couple in sorted_gradients_list:
                    i = i_j_couple[0]
                    j = i_j_couple[1]
                    grad_norms = {name: [] for name, param in common_model.named_parameters()}
                    sorted_sum_gradients.append(sum_gradients[i])
                    sorted_cat_gradients.append(cat_gradients[i])
                    if cross_channel: 
                        if i == new_i : continue
                    if cross_temporal:
                        if j == new_j : continue
                    logger.info(f'domain {i} start training...')
                    train_loader = train_loaders[i][j]
                    val_loader = val_loaders[i][j]
                    test_loader = test_loaders[i][j]
                    cur_mean = torch.from_numpy(cur_means[i][j]).float().to(device)
                    cur_std = torch.from_numpy(cur_stds[i][j]).float().to(device)
                    difference_value = sum_gradients[i][j]
                    p_common = my_sigmoid(p0, difference_value, d_max)
                    common_model.dropout = p_common
                    optimizer = torch.optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)
                    tr = train_one_city(train_loader, common_model, optimizer, epochs=1)
                    # logger.info(f'common_layer_changes on subject_{i+1}: {common_layer_changes[i]}')
                    # logger.info(f'common_shift_changes on subject_{i+1}: {common_shift_changes[i]}')
        num_circles = 3
        sorted_sum_gradients = []
        sorted_cat_gradients = []
        common_stage_num_epochs = 50
        common_learning_rate = 0.001
        weight_decay_common = 0.01
        # total_params = sum(param.nelement() for param in common_model.parameters())
        total_params = len(cat_gradient)
        action_scale_map = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] # Action 0-8
        state_dim = 16
        dqn_agent = DQNAgent(grad_dim=total_params, encoding_dim=in_dim, data_dim=in_dim, state_dim=state_dim, action_dim=len(action_scale_map), device=device, logger=logger).to(device) # (32x37 and 39x16)
        rl_optimizer = optim.Adam(dqn_agent.q_net.parameters(), lr=0.001)
        H_matrix = {name: 0.0 for name, p in common_model.named_parameters()}
        test_acc = 0
        common_model.p = p0 
        for circle_index in range(num_circles):
            for cluster_index in sorted_clusters_gradients_list:
                cluster = cluster_members[cluster_index]  # domains in this cluster
                temp_gradients_list = []
                for i in range(len(cluster)):
                    temp_gradients_list.append(sum_gradients[cluster[i][0]][cluster[i][1]])
                temp_sorted_gradients_list = sort_by_original_gradient(sum_gradients=temp_gradients_list)
                sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
                logger.info(f'sorted_gradients_list: {sorted_gradients_list}')
                for i_j_couple in sorted_gradients_list:
                    i = i_j_couple[0]
                    j = i_j_couple[1]
                    grad_norms = {name: [] for name, param in common_model.named_parameters()}
                    sorted_sum_gradients.append(sum_gradients[i][j])
                    sorted_cat_gradients.append(cat_gradients[i][j])
                    if cross_channel: 
                        if i == new_i : continue
                    if cross_temporal:
                        if j == new_j : continue
                    logger.info(f'domain {i} start training...')
                    train_loader = train_loaders[i][j]
                    val_loader = val_loaders[i][j]
                    test_loader = test_loaders[i][j]
                    cur_mean = torch.from_numpy(cur_means[i][j]).float().to(device)
                    cur_std = torch.from_numpy(cur_stds[i][j]).float().to(device)
                    difference_value = sum_gradients[i][j]
                    rmse, mae, mape, csi, pod, far, common_model, history = train_model_RL(
                        common_model, train_loader, val_loader, test_loader, 
                        exp_model, logger, cur_mean, cur_std, dqn_agent, action_scale_map, H_matrix, rl_optimizer, batch_size=batch_size, i=i, j=j, common=True, n_epochs=1, acc=mape/100, new_domain=False
                    )
                    # logger.info(f'common_layer_changes on subject_{i+1}: {common_layer_changes[i]}')
                    # logger.info(f'common_shift_changes on subject_{i+1}: {common_shift_changes[i]}')
        train_loader = train_loaders[new_i][new_j]
        val_loader = val_loaders[new_i][new_j]
        test_loader = test_loaders[new_i][new_j]
        rmse, mae, mape, csi, pod, far, common_model, history = train_model_RL(
                        common_model, train_loader, val_loader, test_loader, 
                        exp_model, logger, cur_mean, cur_std, dqn_agent, action_scale_map, H_matrix, rl_optimizer, batch_size=batch_size, i=new_i, j=new_j, common=True, n_epochs=1, acc=mape/100, new_domain=True
                    )
        all_test_rmse.append(rmse)
        all_test_mae.append(mae)
        all_test_mape.append(mape)
        history['visualization_saved'] = True
        save_experiment_results(
            result_dir=result_dir,
            experiment_idx=exp_idx + 1,
            dataset_name=dataset,
            backbone_name=exp_model,
            target_subject=[new_i + 1, new_j + 1],
            model_name='BrainAI',
            test_mae=mae,
            history=history,
            logger=logger
        )
        logger.info(f"=================== Experiment {exp_idx+1}/{exp_repeat} results ===================")
        logger.info(f'RMSE: {rmse}, MAE: {mae}, MAPE: {mape}')
        for i in range(len(train_loaders)):
            for j in range(len(train_loaders[0])):
                if i == new_i and j == new_j : continue
                train_loader = train_loaders[i][j]
                val_loader = val_loaders[i][j]
                test_loader = test_loaders[i][j]
                logger.info(f'======================== Validate on domain [{i},{j}] ========================')
                rmse, mae, mape, csi, pod, far = test_model(test_loader=test_loader, model=common_model)
                logger.info(f'RMSE: {rmse}, MAE: {mae}, MAPE: {mape}')
    mean_rmse = sum(all_test_rmse) / len(all_test_rmse)
    mean_mae = sum(all_test_mae) / len(all_test_mae)
    mean_mape = sum(all_test_mape) / len(all_test_mape)
    logger.info(f"RMSE : {all_test_rmse}, mean: {round(mean_rmse, 5)}")
    logger.info(f"MAE : {all_test_mae}, mean: {round(mean_mae, 5)}")
    logger.info(f"MAPE : {all_test_mape}, mean: {round(mean_mape, 5)}")
    if save_npy and len(all_domain_results) > 0:
        logger.info("\n" + "="*60)
        logger.info("Start generating Kepler.gl visualization maps...")
        logger.info("="*60)
        
        new_domain_dir = os.path.join(
            results_dir,
            f'{exp_model}_dataset{dataset_num}_metero{new_i}_month{new_j}_{now}'
        )
        if os.path.exists(new_domain_dir):
            logger.info(f"Generating visualization for new domain [{new_i}, {new_j}]...")
            try:
                map_obj = visualize_predictions_kepler(
                    result_dir=new_domain_dir,
                    station_coords_file=STATION_COORDS_FILE,
                    pm25_mean=pm25_mean,
                    pm25_std=pm25_std,
                    hist_len=hist_len,
                    pred_len=pred_len,
                    output_dir=os.path.join(VISUALIZATION_OUTPUT_DIR, f'new_domain_{new_i}_{new_j}'),
                    save_html=True,
                    save_csv=True
                )
                if map_obj is not None:
                    logger.info("New domain visualization map generated")
                else:
                    logger.warning("New domain visualization map generation failed")
            except Exception as e:
                logger.error(f"New domain visualization failed: {e}")

        if len(all_domain_results) >= 3:
            logger.info(f"Generating multi-domain comparison visualization for {len(all_domain_results)} domains...")
            try:
                # Use the first 5 domains for comparison
                sample_domains = all_domain_results[:5]
                map_obj = visualize_domain_comparison(
                    all_results=sample_domains,
                    station_coords_file=STATION_COORDS_FILE,
                    pm25_mean=pm25_mean,
                    pm25_std=pm25_std,
                    output_dir=os.path.join(VISUALIZATION_OUTPUT_DIR, 'domain_comparison'),
                    hist_len=hist_len,
                    pred_len=pred_len
                )
                if map_obj is not None:
                    logger.info("Domain comparison visualization map generated")
            except Exception as e:
                logger.error(f"Domain comparison visualization failed: {e}")
        
        logger.info("Visualization generation completed.")
if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception('Exception')




