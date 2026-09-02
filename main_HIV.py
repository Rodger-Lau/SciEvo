import os
import copy
import random
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from rdkit import Chem
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.logging import *
from utils.utils import *
from tools.gradient_compute_HIV import *
from models.GINE_HIV import GIN, GAT
from tools.cluster import *
from model.DQN import *
from utils.args import *


# =========================
# Global config
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.001
weight_decay = 1e-4
dataset_name = "HIV"
exp_model = "GINE_target_fewshot_30"
BOND_FEATURE_DIM = 6
BACKBONE_HIDDEN_DIM = 96
BACKBONE_NUM_LAYERS = 3
BACKBONE_DROPOUT = 0.25
writer_dir = f"Writer/BrainAI/{dataset_name}"
target_train_keep_ratios = [0.30]
use_curriculum_ordering = True
enable_freeze_editing = True
enable_ltp_ltd = True

big_threshold = 0.565
small_threshold = 0.535
LTP_TRIGGER_BUDGET = 2
LTD_TRIGGER_BUDGET = 2
LTP_FORCE_EPOCH = 4
LTD_FORCE_EPOCH = 8


# 不平衡学习配置
LOSS_TYPE = "focal"          # 可选: "weighted_bce", "focal", "bce"
FOCAL_GAMMA = 2.0
MAX_POS_WEIGHT = 15.0
BATCH_SIZE = 8

# Roll back to the first clean GINE baseline for ablation:
# keep GINE + edge_attr, but avoid HIV-specific target-stage tuning.
LOSS_TYPE = "weighted_bce"
MAX_POS_WEIGHT = 10.0

# early stopping 监控指标
# 推荐: auc，如果 val_auc 无效，自动 fallback 到 ap，再 fallback 到 f1/mcc。
EARLY_STOP = 10
MONITOR_PRIMARY = "auc"
MONITOR_FALLBACK = "ap"

# 先切分，再只对 train 做 domain-level 重采样，避免 val/test 泄漏
BALANCE_TRAIN_DOMAIN_SIZE = True

os.makedirs(writer_dir, exist_ok=True)
writer = SummaryWriter(writer_dir)


def get_config():
    parser = create_parser()
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"logs/BrainAI/{dataset_name}/{exp_model}"
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, f"{now}.log")
    logger.info(args)
    return args, logger, now


args, logger, now = get_config()
logger.info(f"device:{device}")
logger.info(f"lr:{lr}, weight decay:{weight_decay}, loss_type:{LOSS_TYPE}")
logger.info(
    "HIV GINE patch active: GINE(hidden=96, layers=3, dropout=0.25) "
    "+ bond edge_attr encoding + isolated processed cache"
)
logger.info(
    "Few-shot patch active: "
    f"target-train keep ratios={target_train_keep_ratios}, "
    "target val/test unchanged"
)
logger.info(
    "Module toggles: "
    f"curriculum={use_curriculum_ordering}, "
    f"freeze_editing={enable_freeze_editing}, "
    f"ltp_ltd={enable_ltp_ltd}"
)
logger.info(
    f"Memory thresholds: ltp_freq>{big_threshold}, ltd_freq<{small_threshold}, "
    f"ltp_budget={LTP_TRIGGER_BUDGET}, ltd_budget={LTD_TRIGGER_BUDGET}"
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset
# =========================

class HIVDataset(InMemoryDataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        self.csv_file = csv_file
        super().__init__(root, transform, pre_transform)
        # PyG 旧版/新版 torch.load 行为可能不同，这里保持你原来的写法
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_file)]

    @property
    def processed_file_names(self):
        return ["data_gine.pt"]

    @staticmethod
    def _bond_features(bond):
        bond_type = bond.GetBondType()
        return [
            float(bond_type == Chem.BondType.SINGLE),
            float(bond_type == Chem.BondType.DOUBLE),
            float(bond_type == Chem.BondType.TRIPLE),
            float(bond_type == Chem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]

    def download(self):
        pass

    def _has_carbonyl(self, mol):
        pattern = Chem.MolFromSmarts("[CX3]=[OX1]")
        return mol.HasSubstructMatch(pattern)

    def _has_halogen(self, mol):
        pattern = Chem.MolFromSmarts("[F,Cl,Br,I]")
        return mol.HasSubstructMatch(pattern)

    @staticmethod
    def _domain_id(has_carbonyl, has_halogen):
        # 与 main 中的 domains 编号保持一致：
        # 0: carbonyl & halogen
        # 1: carbonyl only
        # 2: halogen only
        # 3: neither
        if has_carbonyl and has_halogen:
            return 0
        if has_carbonyl and not has_halogen:
            return 1
        if (not has_carbonyl) and has_halogen:
            return 2
        return 3

    def process(self):
        df = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]))

        if "activity" in df.columns and "HIV_active" not in df.columns:
            df.rename(columns={"activity": "HIV_active"}, inplace=True)

        if "HIV_active" not in df.columns:
            raise ValueError("CSV file must contain 'HIV_active' column")

        data_list = []
        domain_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        label_counts = {0: 0, 1: 0}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Molecules"):
            smiles = row["smiles"]
            label = row["HIV_active"]

            try:
                label = int(label)
            except (ValueError, TypeError):
                continue
            if label not in (0, 1):
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            mol = Chem.AddHs(mol)
            has_carbonyl = self._has_carbonyl(mol)
            has_halogen = self._has_halogen(mol)
            domain_id = self._domain_id(has_carbonyl, has_halogen)
            domain_counts[domain_id] += 1
            label_counts[label] += 1

            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetValence(Chem.ValenceType.IMPLICIT),
                    int(atom.GetIsAromatic()),
                    atom.GetTotalNumHs(),
                ]
                atom_features.append(features)

            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_features = self._bond_features(bond)
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append(bond_features)
                edge_attr.append(bond_features)

            try:
                x = torch.tensor(atom_features, dtype=torch.float)
                if edge_index:
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)
                y = torch.tensor([label], dtype=torch.float)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    domain=torch.tensor([domain_id], dtype=torch.long),
                )
                # 保存为 bool，后面用 robust_bool 兼容 bool/tensor
                data.has_carbonyl = bool(has_carbonyl)
                data.has_halogen = bool(has_halogen)
                data_list.append(data)
            except Exception as e:
                print(f"Error processing SMILES: {smiles}, error: {e}")
                continue

        logger.info("\nDomain Distribution:")
        logger.info(f"Domain 0 (carbonyl & halogen): {domain_counts[0]} molecules")
        logger.info(f"Domain 1 (carbonyl only): {domain_counts[1]} molecules")
        logger.info(f"Domain 2 (halogen only): {domain_counts[2]} molecules")
        logger.info(f"Domain 3 (neither): {domain_counts[3]} molecules")
        logger.info(f"Label Distribution: neg={label_counts[0]}, pos={label_counts[1]}")
        logger.info(f"Total molecules processed: {len(data_list)}/{len(df)}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved {len(data_list)} molecules to {self.processed_paths[0]}")


# =========================
# Utilities
# =========================

def robust_bool(x):
    if isinstance(x, torch.Tensor):
        return bool(x.view(-1)[0].item())
    return bool(x)


def get_domain_id_from_data(data):
    has_carbonyl = robust_bool(data.has_carbonyl)
    has_halogen = robust_bool(data.has_halogen)
    if has_carbonyl and has_halogen:
        return 0
    if has_carbonyl and not has_halogen:
        return 1
    if (not has_carbonyl) and has_halogen:
        return 2
    return 3


def get_labels_from_dataset(ds):
    labels = []
    for i in range(len(ds)):
        labels.append(int(ds[i].y.view(-1)[0].item()))
    return np.asarray(labels, dtype=np.int64)


def log_label_stats(name, ds, logger):
    labels = get_labels_from_dataset(ds)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    total = len(labels)
    logger.info(
        f"{name}: total={total}, pos={pos}, neg={neg}, "
        f"pos_ratio={pos / max(total, 1):.6f}"
    )


def safe_train_val_test_split(local_indices, local_labels, seed=42):
    local_indices = np.asarray(local_indices)
    local_labels = np.asarray(local_labels)

    if len(local_indices) < 3:
        # 极端情况：数据过少，尽量不要崩
        return local_indices.tolist(), local_indices.tolist(), local_indices.tolist()

    unique, counts = np.unique(local_labels, return_counts=True)
    can_stratify = (len(unique) == 2) and np.all(counts >= 2)
    stratify_1 = local_labels if can_stratify else None

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        local_indices,
        local_labels,
        test_size=0.3,
        random_state=seed,
        shuffle=True,
        stratify=stratify_1,
    )

    unique_t, counts_t = np.unique(temp_y, return_counts=True)
    can_stratify_temp = (len(unique_t) == 2) and np.all(counts_t >= 2)
    stratify_2 = temp_y if can_stratify_temp else None

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 / 3.0,
        random_state=seed,
        shuffle=True,
        stratify=stratify_2,
    )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def oversample_to_length(indices, target_len, seed=42):
    if len(indices) == 0:
        return []
    rng = np.random.default_rng(seed)
    indices = list(indices)
    if len(indices) >= target_len:
        return indices
    extra = rng.choice(indices, size=target_len - len(indices), replace=True).tolist()
    return indices + extra


def stratified_subsample_indices(indices, dataset_like, keep_ratio, seed=42):
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


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )

        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1.0 - prob)
        focal_factor = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)
        loss = focal_factor * bce

        if self.alpha is not None:
            alpha_pos = torch.tensor(self.alpha, device=targets.device, dtype=targets.dtype)
            alpha_neg = torch.tensor(1.0 - self.alpha, device=targets.device, dtype=targets.dtype)
            alpha_t = torch.where(targets == 1, alpha_pos, alpha_neg)
            loss = alpha_t * loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def make_criterion_from_dataset(ds, device, loss_type=LOSS_TYPE, gamma=FOCAL_GAMMA, max_pos_weight=MAX_POS_WEIGHT, logger=None, prefix=""):
    labels = get_labels_from_dataset(ds)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())

    if pos <= 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = neg / max(pos, 1)
        pos_weight_value = min(float(pos_weight_value), float(max_pos_weight))

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    if logger is not None:
        logger.info(
            f"{prefix} criterion: loss_type={loss_type}, pos={pos}, neg={neg}, "
            f"pos_weight={pos_weight_value:.4f}, focal_gamma={gamma}"
        )

    if loss_type == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_type == "focal":
        return FocalLossWithLogits(gamma=gamma, pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()



def train(model, loader, optimizer, criterion, device, last_out=None, last_param=0.0, epoch=0, mode="acc"):
    model.train()
    total_loss = 0.0
    total = 0
    activate_num = 0
    total_num = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out, activate_num = model(data, activate_num)
        out = out.view(-1)
        target = data.y.view(-1).float()

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total += data.num_graphs
        total_num += 1

    return total_loss / max(total, 1), activate_num / max(total_num, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, logger=None, prefix=""):
    model.eval()
    total_loss = 0.0
    y_true_list, y_prob_list = [], []

    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        out = out.view(-1)
        target = data.y.view(-1).float()

        loss = criterion(out, target)
        total_loss += loss.item() * data.num_graphs

        prob = torch.sigmoid(out)
        y_true_list.append(target.detach().cpu())
        y_prob_list.append(prob.detach().cpu())

    if len(y_true_list) == 0:
        y_true = np.asarray([], dtype=np.int64)
        y_prob = np.asarray([], dtype=np.float32)
    else:
        y_true = torch.cat(y_true_list).numpy().reshape(-1).astype(int)
        y_prob = torch.cat(y_prob_list).numpy().reshape(-1)

    results = {}
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())

    if len(y_prob) > 0:
        prob_min = float(np.nanmin(y_prob))
        prob_max = float(np.nanmax(y_prob))
        prob_mean = float(np.nanmean(y_prob))
        prob_std = float(np.nanstd(y_prob))
    else:
        prob_min = prob_max = prob_mean = prob_std = float("nan")

    auc_valid = True
    auc_reason = "ok"

    if len(y_true) == 0:
        auc_valid = False
        auc_reason = "empty_eval_set"
        auc = 0.5
    elif not np.all(np.isfinite(y_prob)):
        auc_valid = False
        auc_reason = "nan_or_inf_in_y_prob"
        auc = 0.5
    elif len(np.unique(y_true)) < 2:
        auc_valid = False
        auc_reason = "single_class_y_true"
        auc = 0.5
    elif np.nanstd(y_prob) < 1e-12:
        auc_valid = False
        auc_reason = "constant_y_prob"
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception as e:
            auc_valid = False
            auc_reason = f"auc_exception:{repr(e)}"
            auc = 0.5

    safe_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_pred = (safe_prob >= threshold).astype(int)

    results["auc"] = float(auc)
    results["auc_valid"] = bool(auc_valid)
    results["auc_reason"] = auc_reason
    results["pos"] = pos
    results["neg"] = neg
    results["label_pos_rate"] = float(pos / max(len(y_true), 1)) if len(y_true) > 0 else 0.0
    results["prob_min"] = prob_min
    results["prob_max"] = prob_max
    results["prob_mean"] = prob_mean
    results["prob_std"] = prob_std
    results["pred_pos_rate"] = float(y_pred.mean()) if len(y_pred) > 0 else 0.0

    if len(y_true) == 0:
        results["accuracy"] = 0.0
        results["balanced_accuracy"] = 0.0
        results["precision"] = 0.0
        results["recall"] = 0.0
        results["f1"] = 0.0
        results["mcc"] = 0.0
        results["ap"] = 0.0
        results["confusion_matrix"] = None
    else:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
        results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        results["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        results["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        results["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            results["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            results["mcc"] = 0.0
        try:
            results["ap"] = float(average_precision_score(y_true, safe_prob)) if pos > 0 else 0.0
        except Exception:
            results["ap"] = 0.0

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results["confusion_matrix"] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

    if logger is not None:
        logger.info(
            f"{prefix} eval: pos={pos}, neg={neg}, "
            f"prob_min={prob_min:.6f}, prob_max={prob_max:.6f}, "
            f"prob_mean={prob_mean:.6f}, prob_std={prob_std:.6f}, "
            f"auc={results['auc']:.4f}, auc_valid={auc_valid}, reason={auc_reason}, "
            f"acc={results['accuracy']:.4f}, bacc={results['balanced_accuracy']:.4f}, "
            f"label_pos={results['label_pos_rate']:.4f}, pred_pos={results['pred_pos_rate']:.4f}, "
            f"precision={results['precision']:.4f}, recall={results['recall']:.4f}, "
            f"f1={results['f1']:.4f}, mcc={results['mcc']:.4f}, ap={results['ap']:.4f}"
        )

    return total_loss / max(len(loader.dataset), 1), results, y_prob


def metric_for_selection(metrics):
    if MONITOR_PRIMARY == "auc" and metrics.get("auc_valid", False):
        return float(metrics["auc"]), "auc"
    if MONITOR_FALLBACK in metrics:
        return float(metrics[MONITOR_FALLBACK]), MONITOR_FALLBACK
    if "mcc" in metrics:
        return float(metrics["mcc"]), "mcc"
    return float(metrics.get("f1", 0.0)), "f1"


def print_evaluation_metrics(metrics, prefix=""):
    print(f"{prefix}AUC: {metrics['auc']:.4f} valid={metrics.get('auc_valid')} reason={metrics.get('auc_reason')}")
    print(f"{prefix}AP: {metrics.get('ap', 0.0):.4f}")
    print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}Balanced Accuracy: {metrics.get('balanced_accuracy', 0.0):.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall: {metrics['recall']:.4f}")
    print(f"{prefix}F1 Score: {metrics['f1']:.4f}")
    print(f"{prefix}MCC: {metrics.get('mcc', 0.0):.4f}")

    if metrics.get("confusion_matrix"):
        cm = metrics["confusion_matrix"]
        print(f"{prefix}Confusion Matrix:")
        print(f"{prefix}  True Positives: {cm['true_positive']}")
        print(f"{prefix}  True Negatives: {cm['true_negative']}")
        print(f"{prefix}  False Positives: {cm['false_positive']}")
        print(f"{prefix}  False Negatives: {cm['false_negative']}")


# =========================
# Domain statistics
# =========================

def compute_domain_node_statistics(data_list, device="cpu"):
    domain_x = {0: [], 1: [], 2: [], 3: []}

    feature_dim = None
    for data in data_list:
        did = get_domain_id_from_data(data)
        domain_x[did].append(data.x.to(device))
        feature_dim = data.x.shape[1]

    if feature_dim is None:
        raise ValueError("Empty dataset, cannot compute domain statistics.")

    cur_means, cur_stds = [], []
    domain_names = [
        "Carbonyl & Halogen",
        "Carbonyl only",
        "Halogen only",
        "Neither",
    ]

    for domain_id in range(4):
        if len(domain_x[domain_id]) == 0:
            cur_means.append(torch.zeros(feature_dim, device=device))
            cur_stds.append(torch.ones(feature_dim, device=device))
            logger.info(f"  Domain {domain_id} ({domain_names[domain_id]}): no data")
            continue

        all_x = torch.cat(domain_x[domain_id], dim=0)
        mean = all_x.mean(dim=0)
        std = all_x.std(dim=0).clamp(min=1e-6)
        cur_means.append(mean)
        cur_stds.append(std)

        logger.info(
            f"  Domain {domain_id} ({domain_names[domain_id]}): "
            f"{len(domain_x[domain_id])} graphs, {all_x.shape[0]} nodes, "
            f"mean={mean}, std={std}"
        )

    return cur_means, cur_stds
def train_model_RL(
    model,
    train_loader,
    val_loader,
    test_loader,
    logger,
    cur_mean,
    cur_std,
    dqn_agent,
    action_scale_map,
    H_matrix,
    rl_optimizer,
    batch_size,
    c,
    write=False,
    common=True,
    n_epochs=50,
    new_domain=False,
    acc=0,
):
    model = model.to(device)
    rho = 0.001 * acc
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = make_criterion_from_dataset(
        train_loader.dataset,
        device,
        loss_type=LOSS_TYPE,
        logger=logger,
        prefix=f"[Domain {c}] train_model_RL",
    )

    current_state_vec = None
    activate_freq = 0.5
    best_score = -1e9
    best_state = None
    best_epoch = -1
    cnt = 0
    test_metrics = None
    ltp_count = 0
    ltd_count = 0

    for epoch in range(n_epochs):
        if common and write:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f"grad_norm/{name}", param.grad.norm().item(), epoch)

        prev_params = {name: param.data.clone() for name, param in model.named_parameters()}

        # 构造 DQN state：梯度 + 当前 domain 统计
        if current_state_vec is None:
            model.eval()
            for param in model.parameters():
                param.requires_grad = True

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out, _ = model(data, 0)
                out = out.view(-1)
                target = data.y.view(-1).float()
                loss = criterion(out, target)
                loss.backward()
                break
            grad_flat = get_flat_grad(model).detach()
        else:
            grad_flat = current_state_vec[0]

        grad_input = grad_flat.unsqueeze(0)
        mean_input = cur_mean.unsqueeze(0)
        std_input = cur_std.unsqueeze(0)
        logger.info(
            f"grad_input shape:{grad_input.shape}, "
            f"mean_input shape:{mean_input.shape}, std_input shape:{std_input.shape}"
        )

        dqn_state = (grad_input, mean_input, std_input)
        action_idx = dqn_agent.select_action(dqn_state, new_domain)
        current_state_vec = (grad_input.squeeze(0), mean_input.squeeze(0), std_input.squeeze(0))

        scale = action_scale_map[action_idx]
        if not new_domain and enable_ltp_ltd:
            logger.info(f"activate freq:{activate_freq}")
            if activate_freq > big_threshold and ltp_count < LTP_TRIGGER_BUDGET:
                logger.info("高激活频率，LTP")
                scale *= 0.9
                ltp_count += 1
            elif activate_freq < small_threshold and ltd_count < LTD_TRIGGER_BUDGET:
                logger.info("低激活频率，LTD")
                scale /= 0.9
                ltd_count += 1
            elif epoch + 1 >= LTP_FORCE_EPOCH and ltp_count == 0:
                logger.info("保底触发，LTP")
                scale *= 0.9
                ltp_count += 1
            elif epoch + 1 >= LTD_FORCE_EPOCH and ltd_count == 0:
                logger.info("保底触发，LTD")
                scale /= 0.9
                ltd_count += 1

        logger.info(f"freeze scale:{scale}")
        if not new_domain and enable_ltp_ltd:
            logger.info(f"ltp/ltd events:{ltp_count}/{LTP_TRIGGER_BUDGET}, {ltd_count}/{LTD_TRIGGER_BUDGET}")
        sorted_layers = sorted(H_matrix.items(), key=lambda item: item[1])
        total_layers = len(sorted_layers)
        freeze_count = int(total_layers * scale) if enable_freeze_editing else 0
        layers_to_freeze = [name for name, _ in sorted_layers[:freeze_count]]

        model.train()
        for param in model.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            param.requires_grad = name not in layers_to_freeze

        epoch_loss = 0.0
        epoch_graphs = 0
        activate_num = 0
        batch_num = 0
        correct = 0
        total_pred = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out, activate_num = model(data, activate_num)
            out = out.view(-1)
            y = data.y.view(-1).float()
            if rho > 0:
                soft_prob = torch.sigmoid(out.detach())
                target = ((1.0 - rho) * y + rho * soft_prob).clamp(0.0, 1.0)
            else:
                target = y

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * data.num_graphs
            epoch_graphs += data.num_graphs
            batch_num += 1

            pred = (torch.sigmoid(out.detach()) >= 0.5).float()
            correct += (pred.cpu() == y.detach().cpu()).sum().item()
            total_pred += y.numel()

        epoch_loss = epoch_loss / max(epoch_graphs, 1)
        activate_freq = activate_num / max(batch_num, 1)
        train_acc = correct / max(total_pred, 1)

        updates = calculate_layer_updates(model, prev_params)
        for name, update_val in updates.items():
            if name not in layers_to_freeze:
                H_matrix[name] = 0.9 * H_matrix[name] + 0.1 * update_val

        if not new_domain:
            model.eval()
            for param in model.parameters():
                param.requires_grad = True

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out, _ = model(data, 0)
                out = out.view(-1)
                target = data.y.view(-1).float()
                loss = criterion(out, target)
                loss.backward()
                break

            grad_flat_next = get_flat_grad(model).detach()
            next_state_vec = (grad_flat_next, cur_mean, cur_std)
            reward = 1.0 / max(epoch_loss, 1e-8)
            dqn_agent.memory.append((current_state_vec, action_idx, reward, next_state_vec))
            dqn_agent.train_step(batch_size=batch_size, optimizer=rl_optimizer)
            current_state_vec = copy.deepcopy(next_state_vec)
        val_loss, val_metrics, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            logger=logger,
            prefix=f"[Domain {c}] Val",
        )
        val_score, monitor_name = metric_for_selection(val_metrics)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            cnt = 0
        else:
            cnt += 1

        logger.info(
            f"  [Domain {c:3d}] Ep {epoch + 1:3d}/{n_epochs} "
            f"train_loss={epoch_loss:.6f} train_acc={train_acc:.4f} "
            f"val_{monitor_name}={val_score:.4f} best_score={best_score:.4f} "
            f"best_epoch={best_epoch} patience={cnt}/{EARLY_STOP}"
        )

        if cnt >= EARLY_STOP:
            logger.info(f"  [Domain {c:3d}] → Early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics, _ = evaluate(
        model,
        test_loader,
        criterion,
        device,
        logger=logger,
        prefix=f"[Domain {c}] Test",
    )

    test_auc = test_metrics["auc"]
    test_acc = test_metrics["accuracy"]
    logger.info(
        f"\n  >>> [Domain {c:3d}] "
        f"AUC={test_auc:.4f} ACC={test_acc:.4f} "
        f"BACC={test_metrics['balanced_accuracy']:.4f} "
        f"F1={test_metrics['f1']:.4f} MCC={test_metrics['mcc']:.4f} "
        f"AP={test_metrics['ap']:.4f} "
        f"auc_valid={test_metrics['auc_valid']} reason={test_metrics['auc_reason']}"
    )
    return test_auc, test_acc, model
def main():
    setup_seed(42)
    p0 = BACKBONE_DROPOUT
    batch_size = BATCH_SIZE

    data_dir = "./data/hiv"
    src_csv = "./data/hiv/HIV.csv"
    dataset = HIVDataset(root=data_dir, csv_file=src_csv)
    logger.info(f"数据集包含 {len(dataset)} 个图")

    domains = {0: [], 1: [], 2: [], 3: []}
    domain_labels = {0: [], 1: [], 2: [], 3: []}

    for idx in range(len(dataset)):
        data = dataset[idx]
        did = get_domain_id_from_data(data)
        domains[did].append(idx)
        domain_labels[did].append(int(data.y.view(-1)[0].item()))

    logger.info("================ Raw domain/label distribution ================")
    for did in range(4):
        labels = np.asarray(domain_labels[did], dtype=np.int64)
        pos = int((labels == 1).sum())
        neg = int((labels == 0).sum())
        logger.info(f"Domain {did}: total={len(labels)}, pos={pos}, neg={neg}, pos_ratio={pos / max(len(labels), 1):.6f}")
    first_stage_num_epochs = 10
    logger.info("计算各域节点层面均值和标准差:")
    cur_means, cur_stds = compute_domain_node_statistics(list(dataset), device="cpu")

    split_indices = {}
    max_train_len = 0
    for did in range(4):
        train_idx, val_idx, test_idx = safe_train_val_test_split(domains[did], domain_labels[did], seed=42)
        split_indices[did] = {"train": train_idx, "val": val_idx, "test": test_idx}
        max_train_len = max(max_train_len, len(train_idx))

    train_loaders, val_loaders, test_loaders = [], [], []
    train_datasets, val_datasets, test_datasets = [], [], []
    raw_train_split_indices = {}
    raw_val_split_indices = {}
    raw_test_split_indices = {}

    for did in range(4):
        train_idx = split_indices[did]["train"]
        val_idx = split_indices[did]["val"]
        test_idx = split_indices[did]["test"]
        raw_train_split_indices[did] = list(train_idx)
        raw_val_split_indices[did] = list(val_idx)
        raw_test_split_indices[did] = list(test_idx)

        if BALANCE_TRAIN_DOMAIN_SIZE:
            train_idx_balanced = oversample_to_length(train_idx, max_train_len, seed=42 + did)
        else:
            train_idx_balanced = train_idx

        train_dataset = dataset[train_idx_balanced]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

        log_label_stats(f"Domain {did} Train", train_dataset, logger)
        log_label_stats(f"Domain {did} Val", val_dataset, logger)
        log_label_stats(f"Domain {did} Test", test_dataset, logger)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    exp_repeat = 3
    new_i = 3
    logger.info(f"new_i: {new_i}")

    sample_data = dataset[0]
    in_channels = sample_data.x.shape[1]
    fewshot_results = {}

    for keep_ratio in target_train_keep_ratios:
        ratio_pct = int(round(keep_ratio * 100))
        logger.info("\n" + "#" * 120)
        logger.info(
            f"Few-shot target-train start: keep_ratio={keep_ratio:.2f} ({ratio_pct}%), "
            "target val/test unchanged"
        )
        logger.info("#" * 120)

        test_aucs = []
        test_accs = []

        for exp_idx in range(exp_repeat):
            exp_seed = 42 + exp_idx
            setup_seed(exp_seed)
            logger.info(
                f"\n{'=' * 60}  Target train {ratio_pct}% | Experiment {exp_idx + 1}/{exp_repeat}  {'=' * 60}"
            )

            sum_gradients = [None] * 4
            cat_grad_dict = {}

            for key in range(4):
                if key == new_i:
                    sum_gradients[key] = torch.tensor(0.0, device=device)
                    continue
    
                train_loader = train_loaders[key]
                model = GIN(
                    in_channels=in_channels,
                    hidden_dim=BACKBONE_HIDDEN_DIM,
                    num_layers=BACKBONE_NUM_LAYERS,
                    dropout=BACKBONE_DROPOUT,
                ).to(device)
    
                criterion = make_criterion_from_dataset(
                    train_loader.dataset,
                    device,
                    loss_type=LOSS_TYPE,
                    logger=logger,
                    prefix=f"[Stage1 Domain {key}]",
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
                logger.info(f"{key}开始训练...")
                for epoch in range(1, first_stage_num_epochs + 1):
                    train_loss, act_freq = train(model, train_loader, optimizer, criterion, device, epoch=epoch)
                    logger.info(f"[Stage1 Domain {key}] epoch={epoch} train_loss={train_loss:.6f} act_freq={act_freq:.4f}")
    
                sum_grad, cat_grad = compute_gradient(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    criterion=criterion,
                )
                sum_gradients[key] = sum_grad
                cat_grad_dict[key] = cat_grad
                logger.info(f"[Stage1 Domain {key}] sum_grad={sum_grad}")
            n_clusters = 2
            cluster_labels, cluster_members = clustering_dict(n_clusters=n_clusters, data_dict=cat_grad_dict, algorithm="kmeans")
            logger.info("聚类结果为：")
            cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_grad_dict)
            logger.info(cluster_grad)
    
            common_model = GIN(
                in_channels=in_channels,
                hidden_dim=BACKBONE_HIDDEN_DIM,
                num_layers=BACKBONE_NUM_LAYERS,
                dropout=BACKBONE_DROPOUT,
            ).to(device)
    
            sum_gradients_cluster = [sum(cluster_grad[i]) for i in range(len(cluster_grad))]
            if use_curriculum_ordering:
                sorted_clusters_gradients_list = sort_by_original_gradient(sum_gradients=sum_gradients_cluster)
            else:
                sorted_clusters_gradients_list = list(cluster_members.keys())
            second_stage_circle = 1
            second_stage_num_epochs = 10
    
            for _ in range(second_stage_circle):
                for cluster_index in sorted_clusters_gradients_list:
                    train_number = 1
                    cluster = cluster_members[cluster_index]
                    if use_curriculum_ordering:
                        temp_sorted_gradients_list = sort_by_original_gradient(
                            sum_gradients=[sum_gradients[domain] for domain in cluster]
                        )
                        sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
                    else:
                        sorted_gradients_list = list(cluster)
                    logger.info(sorted_gradients_list)
    
                    for key in sorted_gradients_list:
                        if key == new_i:
                            continue
                        train_number -= 1
                        if train_number < 0:
                            continue
    
                        train_loader = train_loaders[key]
                        val_loader = val_loaders[key]
                        test_loader = test_loaders[key]
    
                        criterion = make_criterion_from_dataset(
                            train_loader.dataset,
                            device,
                            loss_type=LOSS_TYPE,
                            logger=logger,
                            prefix=f"[Stage2 Domain {key}]",
                        )
                        optimizer = torch.optim.Adam(common_model.parameters(), lr=lr, weight_decay=weight_decay)
                        common_model.drop_out = p0
    
                        logger.info(f"{key}开始训练...")
                        for epoch in range(1, second_stage_num_epochs + 1):
                            train_loss, act_freq = train(common_model, train_loader, optimizer, criterion, device, epoch=epoch)
                            val_loss, val_metrics, _ = evaluate(
                                common_model,
                                val_loader,
                                criterion,
                                device,
                                logger=logger,
                                prefix=f"[Stage2 Domain {key}] Val",
                            )
                            logger.info(
                                f"[Stage2 Domain {key}] epoch={epoch} train_loss={train_loss:.6f} "
                                f"val_auc={val_metrics['auc']:.4f} valid={val_metrics['auc_valid']} "
                                f"val_ap={val_metrics['ap']:.4f} val_f1={val_metrics['f1']:.4f}"
                            )
    
                        test_loss, test_metrics, _ = evaluate(
                            common_model,
                            test_loader,
                            criterion,
                            device,
                            logger=logger,
                            prefix=f"[Stage2 Domain {key}] Test",
                        )
                        print_evaluation_metrics(test_metrics, prefix=f"[Stage2 Domain {key}] ")
            third_stage_circle = 3
            third_stage_num_epochs = 100
            total_params = len(cat_grad)
            action_scale_map = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            state_dim = 16
            in_dim = len(cur_means[0])
    
            dqn_agent = DQNAgent(
                grad_dim=total_params,
                encoding_dim=in_dim,
                data_dim=in_dim,
                state_dim=state_dim,
                action_dim=len(action_scale_map),
                device=device,
                logger=logger,
            ).to(device)
            rl_optimizer = optim.Adam(dqn_agent.q_net.parameters(), lr=0.001)
            H_matrix = {name: 0.0 for name, p in common_model.named_parameters()}
    
            for _ in range(third_stage_circle):
                for cluster_index in sorted_clusters_gradients_list:
                    train_number = 1
                    cluster = cluster_members[cluster_index]
                    if use_curriculum_ordering:
                        temp_sorted_gradients_list = sort_by_original_gradient(
                            sum_gradients=[sum_gradients[domain] for domain in cluster]
                        )
                        sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
                    else:
                        sorted_gradients_list = list(cluster)
                    logger.info(sorted_gradients_list)
    
                    for key in sorted_gradients_list:
                        if key == new_i:
                            continue
                        train_number -= 1
                        if train_number < 0:
                            continue
    
                        logger.info(f"{key}开始训练...")
                        common_model.drop_out = p0
                        cur_mean = cur_means[key].to(device)
                        cur_std = cur_stds[key].to(device)
    
                        test_auc, test_acc, common_model = train_model_RL(
                            common_model,
                            train_loaders[key],
                            val_loaders[key],
                            test_loaders[key],
                            logger,
                            cur_mean,
                            cur_std,
                            dqn_agent,
                            action_scale_map,
                            H_matrix,
                            rl_optimizer,
                            batch_size,
                            c=key,
                            write=False,
                            common=True,
                            n_epochs=third_stage_num_epochs,
                            new_domain=False,
                            acc=0.0,
                        )
            key = new_i
            fourth_stage_num_epochs = 200
            cur_mean = cur_means[new_i].to(device)
            cur_std = cur_stds[new_i].to(device)

            target_train_indices = stratified_subsample_indices(
                raw_train_split_indices[new_i],
                dataset,
                keep_ratio=keep_ratio,
                seed=exp_seed
            )
            target_val_indices = raw_val_split_indices[new_i]
            target_test_indices = raw_test_split_indices[new_i]
            target_train_dataset = dataset[target_train_indices]
            target_val_dataset = dataset[target_val_indices]
            target_test_dataset = dataset[target_test_indices]
            target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            target_val_loader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

            logger.info(
                f"[FewShot Target Domain {new_i}] keep_ratio={keep_ratio:.2f} ({ratio_pct}%), "
                f"train={len(target_train_indices)}, val={len(target_val_indices)}, test={len(target_test_indices)}"
            )
            log_label_stats(f"[FewShot Target {new_i}] Train", target_train_dataset, logger)
            log_label_stats(f"[FewShot Target {new_i}] Val", target_val_dataset, logger)
            log_label_stats(f"[FewShot Target {new_i}] Test", target_test_dataset, logger)
    
            test_auc, test_acc, common_model = train_model_RL(
                common_model,
                target_train_loader,
                target_val_loader,
                target_test_loader,
                logger,
                cur_mean,
                cur_std,
                dqn_agent,
                action_scale_map,
                H_matrix,
                rl_optimizer,
                batch_size,
                c=key,
                write=False,
                common=False,
                n_epochs=fourth_stage_num_epochs,
                new_domain=True,
                acc=0.0,
            )
    
            logger.info(f"第{exp_idx + 1}次实验结果为： test_auc: {test_auc}, test_acc: {test_acc}")
            test_aucs.append(test_auc)
            test_accs.append(test_acc)

        mean_auc = sum(test_aucs) / max(len(test_aucs), 1)
        mean_acc = sum(test_accs) / max(len(test_accs), 1)
        fewshot_results[keep_ratio] = {
            "aucs": list(test_aucs),
            "accs": list(test_accs),
            "mean_auc": mean_auc,
            "mean_acc": mean_acc,
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
    logger.info("Few-shot target-train summary (val/test unchanged)")
    for keep_ratio in target_train_keep_ratios:
        result = fewshot_results[keep_ratio]
        logger.info(
            f"keep_ratio={keep_ratio:.2f} ({int(round(keep_ratio * 100))}%): "
            f"mean_auc={result['mean_auc']:.5f}, mean_acc={result['mean_acc']:.5f}"
        )
    logger.info("=" * 118)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Exception")
        raise
