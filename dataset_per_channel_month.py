"""
KnowAir 按城市划分 Domain 的数据加载模块
==========================================
将原始 HazeData（所有城市混合）按 city idx 拆分为多个独立的 Domain，
每个 Domain 对应一个城市，拥有自己的 DataLoader。

返回:
    train_loaders: List[DataLoader], 长度 = city_num
    val_loaders  : List[DataLoader], 同上
    test_loaders : List[DataLoader], 同上
"""

import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

from util import config, file_dir
from graph import Graph
from dataset import HazeData
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import pandas as pd
from torch.utils import data
from torch.utils.data import Subset
import datetime
def _extract_months(time_arr):
    """提取预测目标所在的月份 (兼容 int 时间戳和 datetime)"""
    pred_time = time_arr[:, -1]
    # if np.issubdtype(pred_time.dtype, np.integer):
    #     return (pred_time % 10000) // 100
    # else:
    #     return np.array([pd.Timestamp(t).month for t in pred_time])
    months = []
    for t in pred_time:
        # ★ 关键修改：使用 datetime 的 utcfromtimestamp，它会将 Unix 时间戳(秒)正确转为日期
        dt = datetime.datetime.fromtimestamp(t)
        months.append(dt.month)
        
    return np.array(months)


def get_channel_month_2d_loaders(graph, hist_len, pred_len, dataset_num,
                                 batch_size, num_workers=0, logger=None):
    """
    二维 Domain 划分：Channel × Month
    返回二维列表 loaders[channel_idx][month_idx]
    每个 Loader 的特征维度 = 5 (1个通道 + 4个辅助特征)
    """
    # ========================================================================= #
    #   1. 获取全局预处理数据                                                    #
    # ========================================================================= #
    train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
    val_data   = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
    test_data  = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')

    train_feat, val_feat, test_feat = train_data.feature, val_data.feature, test_data.feature
    train_pm25, val_pm25, test_pm25 = train_data.pm25, val_data.pm25, test_data.pm25
    train_time, val_time, test_time = train_data.time_arr, val_data.time_arr, test_data.time_arr
    # logger.info(f'train_time:{train_time}, val_time:{val_time}, test_time:{test_time}')

    # ========================================================================= #
    #   2. 全局统计量 & 辅助特征归一化修复                                        #
    # ========================================================================= #
    metero_use = config['experiments']['metero_use']
    metero_num = len(metero_use)
    aux_num    = 4
    aux_start  = metero_num

    pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std
    wind_mean, wind_std = train_data.wind_mean, train_data.wind_std

    # 修复辅助特征未归一化问题
    raw_aux_train = train_feat[:, :, :, aux_start:aux_start + aux_num]
    aux_means = raw_aux_train.mean(axis=(0, 1, 2))
    aux_stds  = raw_aux_train.std(axis=(0, 1, 2))
    aux_stds  = np.where(aux_stds < 1e-5, 1.0, aux_stds)

    train_feat[:, :, :, aux_start:aux_start + aux_num] = (raw_aux_train - aux_means) / aux_stds
    val_feat[:, :, :, aux_start:aux_start + aux_num]   = (val_feat[:, :, :, aux_start:aux_start + aux_num] - aux_means) / aux_stds
    test_feat[:, :, :, aux_start:aux_start + aux_num]  = (test_feat[:, :, :, aux_start:aux_start + aux_num] - aux_means) / aux_stds

    # 提取归一化后的辅助特征
    aux_train = train_feat[:, :, :, aux_start:aux_start + aux_num]
    aux_val   = val_feat[:, :, :, aux_start:aux_start + aux_num]
    aux_test  = test_feat[:, :, :, aux_start:aux_start + aux_num]

    # ========================================================================= #
    #   3. 提取月份索引                                                          #
    # ========================================================================= #
    months_train = _extract_months(train_time)
    months_val   = _extract_months(val_time)
    months_test  = _extract_months(test_time)

    # ========================================================================= #
    #   4. 构建 Channel × Month 二维 Domain Loaders                              #
    # ========================================================================= #
    # 总通道数 = 气象通道数 + 1个PM2.5通道
    total_channels = metero_num + 1 
    domain_names = list(metero_use) + ['PM2.5']

    # 初始化二维数组: shape = (total_channels, 12)
    train_loaders_2d = [[None for _ in range(12)] for _ in range(total_channels)]
    val_loaders_2d   = [[None for _ in range(12)] for _ in range(total_channels)]
    test_loaders_2d  = [[None for _ in range(12)] for _ in range(total_channels)]
    
    # 记录统计量 (为了跨域泛化，同一通道的12个月共享同一套统计量，即全局统计量)
    feature_means_2d = [[None for _ in range(12)] for _ in range(total_channels)]
    feature_stds_2d  = [[None for _ in range(12)] for _ in range(total_channels)]

    logger.info("=" * 120)
    logger.info(f"  2D Domain Loading (Channel × Month) | hist={hist_len}, pred={pred_len}, "
          f"dim_per_domain=5, grid_size=({total_channels}, 12)")
    logger.info("=" * 120)

    for c_idx in range(total_channels):
        # --- 确定当前通道的数据切片 ---
        if c_idx < metero_num:
            # 气象通道
            feat_c_train = train_feat[:, :, :, c_idx:c_idx+1]
            feat_c_val   = val_feat[:, :, :, c_idx:c_idx+1]
            feat_c_test  = test_feat[:, :, :, c_idx:c_idx+1]
            c_mean = train_data.feature_mean[c_idx]
            c_std  = train_data.feature_std[c_idx]
        else:
            # PM2.5 通道
            feat_c_train = train_pm25
            feat_c_val   = val_pm25
            feat_c_test  = test_pm25
            c_mean = train_data.pm25_mean
            c_std  = train_data.pm25_std

        # 当前通道的 6 维统计量: [通道均值(1), 辅助均值(4), pm25均值(1)]
        current_domain_stat_mean = np.concatenate([[c_mean], aux_means, [train_data.pm25_mean]])
        current_domain_stat_std  = np.concatenate([[c_std],  aux_stds,  [train_data.pm25_std]])

        # --- 内层循环：遍历 12 个月 ---
        for m_idx in range(12):
            m = m_idx + 1 # 实际月份 1~12
            
            # 找到当前月份的样本索引
            idx_train = np.where(months_train == m)[0]
            idx_val   = np.where(months_val == m)[0]
            idx_test  = np.where(months_test == m)[0]

            # 填充统计量 (无论是否有数据，都填充该通道的全局统计量)
            feature_means_2d[c_idx][m_idx] = current_domain_stat_mean
            feature_stds_2d[c_idx][m_idx]  = current_domain_stat_std

            if len(idx_train) == 0 and len(idx_val) == 0 and len(idx_test) == 0:
                continue # 保持 None

            # 提取该月数据并拼接: (N_m, seq, city, 5)
            f_train_m = np.concatenate([feat_c_train[idx_train], aux_train[idx_train]], axis=-1)
            f_val_m   = np.concatenate([feat_c_val[idx_val],     aux_val[idx_val]],     axis=-1)
            f_test_m  = np.concatenate([feat_c_test[idx_test],   aux_test[idx_test]],   axis=-1)

            # 构建 DataLoader
            # ★ 警告：二维切分后单格数据量很小，必须 drop_last=False，否则会丢失整个月的最后几个样本！
            def make_loader(feat, pm25, time, shuffle):
                if len(feat) == 0: return None
                ds = TensorDataset(torch.FloatTensor(pm25),
                                   torch.FloatTensor(feat),
                                   torch.LongTensor(time))
                return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, drop_last=False) 

            train_loaders_2d[c_idx][m_idx] = make_loader(f_train_m, train_pm25[idx_train], train_time[idx_train], shuffle=True)
            val_loaders_2d[c_idx][m_idx]   = make_loader(f_val_m,   val_pm25[idx_val],     val_time[idx_val],     shuffle=False)
            test_loaders_2d[c_idx][m_idx]  = make_loader(f_test_m,  test_pm25[idx_test],   test_time[idx_test],   shuffle=False)

        # 打印该通道的信息
        train_counts = [len(train_loaders_2d[c_idx][i].dataset) if train_loaders_2d[c_idx][i] else 0 for i in range(12)]
        print(f"  Ch[{c_idx:2d}] {domain_names[c_idx]:15s} | "
              f"Train samples/month: {train_counts}")

    # ========================================================================= #
    #   5. 返回结果                                                              #
    # ========================================================================= #
    in_dim_per_domain = aux_num + 1 + train_data.pm25.shape[-1]  # 4 + 1 + 1 = 6

    logger.info("-" * 120)
    logger.info(f"  Total Grid: {total_channels} × 12 | in_dim per cell: {in_dim_per_domain}")
    logger.info("=" * 120)

    return (train_loaders_2d, val_loaders_2d, test_loaders_2d,
            feature_means_2d, feature_stds_2d,
            pm25_mean, pm25_std,
            wind_mean, wind_std,
            domain_names, in_dim_per_domain)


def few_shot_sample(loader, ratio, seed=42):
    """
    对单个 DataLoader 的数据集进行随机下采样。
    
    :param loader:   原始的 DataLoader (如 train_loaders[100])
    :param ratio:    保留比例 (如 0.1 表示只保留 10% 的数据)
    :param seed:     随机种子，保证可复现
    :return:         新的 DataLoader
    """
    dataset = loader.dataset
    total_len = len(dataset)
    sample_len = max(1, int(total_len * ratio))
    
    # 随机无放回抽取索引
    np.random.seed(seed)
    sampled_indices = np.random.choice(total_len, size=sample_len, replace=False)
    # 打乱一下索引，防止 Subset 按原顺序取出导致训练时数据分布有偏
    np.random.shuffle(sampled_indices)
    
    # 用 Subset 包装
    sub_dataset = Subset(dataset, sampled_indices)
    
    # 用和原始 loader 完全相同的参数创建新 loader
    new_loader = DataLoader(
        sub_dataset,
        batch_size=loader.batch_size,
        shuffle=True,          # Subset 内部已经打乱，这里设 True 再洗一次更安全
        num_workers=loader.num_workers,
        drop_last=False        # 大尺度窗口每月样本可能较少，保留小 batch
    )
    return new_loader

def apply_few_shot_sampling(loader, few_shot_scale, logger=None):
    """
    对按通道划分的 DataLoader 应用 few-shot 采样。
    从每个 domain 的训练集中随机抽取 shot_num 个样本。
    
    :param train_loaders: 原始的训练集 loaders 列表 (每个里面是 TensorDataset)
    :param val_loaders:   验证集 loaders (不采样，直接返回)
    :param test_loaders:  测试集 loaders (不采样，直接返回)
    :param shot_num:      Few-shot 的样本数量 K
    :return:              采样后的 fs_train_loaders, 以及原始的 val_loaders, test_loaders
    """
    
    
    dataset = loader.dataset  # 这是一个 TensorDataset
    total_samples = len(dataset)
    shot_num = max(1, int(total_samples * few_shot_scale))
        
    # 随机无放回抽取 K 个索引
    indices = np.random.choice(total_samples, size=shot_num, replace=False)
    
    # 使用 Subset 包装原 Dataset
    fs_dataset = Subset(dataset, indices)
    
    # 创建新的 DataLoader，保持原有的 batch_size 等参数
    # 注意：Few-shot 场景下强烈建议 drop_last=False，防止 K 个样本因为凑不够一个 batch 被丢弃
    fs_loader = DataLoader(
        fs_dataset, 
        batch_size=loader.batch_size, 
        shuffle=True,  # 每个 epoch 内部会重新打乱这 K 个样本
        num_workers=loader.num_workers, 
        drop_last=False
    )
    logger.info(f"从 {total_samples} 个样本中 Few-shot 采样了 {shot_num} 个.")
        
    return fs_loader
# --------------------------------------------------------------------------- #
#  1. 单城市数据集
# --------------------------------------------------------------------------- #
class HazeDataPerCity(data.Dataset):
    """
    封装 **单个城市** 的滑动窗口数据。

    Parameters
    ----------
    pm25_city : np.ndarray, shape (N, seq_len, 1)
    feat_city : np.ndarray, shape (N, seq_len, feat_dim)
    time_arr  : np.ndarray, shape (N, seq_len)
    """

    def __init__(self, pm25_city, feat_city, time_arr):
        self.pm25    = torch.from_numpy(pm25_city).float()
        self.feature = torch.from_numpy(feat_city).float()
        self.time_arr = torch.from_numpy(time_arr).float()

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index], self.time_arr[index]


