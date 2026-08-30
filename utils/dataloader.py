import torch
import numpy as np
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta

class StandardScaler:
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def _slice_output(y, output_len):
    if output_len is None:
        return y
    if output_len <= 0:
        raise ValueError(f"output_len must be positive, got {output_len}")
    max_len = y.shape[1]
    if output_len > max_len:
        raise ValueError(
            f"output_len={output_len} exceeds available horizon={max_len}"
        )
    return y[:, :output_len, ...]


def _sample_end_timestamp(sample):
    time_fields = sample[-1, 0, 5:11]
    year, month, day, hour, minute, second = [int(value) for value in time_fields]
    return datetime(year, month, day, hour, minute, second)


def _calendar_group_key(timestamp, pattern):
    if pattern == 'daily_1h':
        return (timestamp.year, timestamp.month, timestamp.day)
    if pattern == 'weekly_1day':
        iso_year, iso_week, _ = timestamp.isocalendar()
        return (iso_year, iso_week)
    if pattern == 'monthly_1week':
        return (timestamp.year, timestamp.month)
    raise ValueError(f'Unsupported calendar pattern: {pattern}')


def _select_fixed_missing_nodes(num_nodes, missing_node_ratio, seed=0):
    if missing_node_ratio is None or missing_node_ratio <= 0:
        return np.array([], dtype=int)

    k = int(round(num_nodes * float(missing_node_ratio)))
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, num_nodes)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(num_nodes), size=k, replace=False))


def _apply_fixed_missing_nodes(data, missing_node_ratio=0.0, seed=0, logger=None):
    num_nodes = data['x_train'].shape[2]
    missing_nodes = _select_fixed_missing_nodes(num_nodes, missing_node_ratio, seed=seed)
    if missing_nodes.size == 0:
        return data, missing_nodes

    data['x_train'][:, :, missing_nodes, :] = 0
    data['x_val'][:, :, missing_nodes, :] = 0
    data['x_test'][:, :, missing_nodes, :] = 0

    if logger is not None:
        logger.info(
            f'Apply fixed spatial missing nodes: ratio={missing_node_ratio}, '
            f'count={len(missing_nodes)}/{num_nodes}, nodes={missing_nodes.tolist()}'
        )

    return data, missing_nodes


def _apply_tail_missing(arr, indices, tail_steps, missing_nodes):
    if tail_steps is None or tail_steps <= 0:
        return
    time_len = arr.shape[1]
    if tail_steps > time_len:
        raise ValueError(f"missing_tail_steps={tail_steps} exceeds input_len={time_len}")
    if missing_nodes is None or len(missing_nodes) == 0:
        arr[indices, -tail_steps:, :, :] = 0
        return
    time_idx = np.arange(time_len - tail_steps, time_len)
    feat_idx = np.arange(arr.shape[3])
    arr[np.ix_(indices, time_idx, missing_nodes, feat_idx)] = 0


def _drop_contiguous_train_time_window(data, missing_time_hours=0, missing_time_mode='random_contiguous', seed=0, logger=None):
    if missing_time_hours is None or missing_time_hours <= 0:
        return data, None

    train_times = [_sample_end_timestamp(data['x_train'][index]) for index in range(len(data['x_train']))]
    unique_times = sorted(set(train_times))
    if not unique_times:
        return data, None

    duration = timedelta(hours=float(missing_time_hours))
    if missing_time_mode not in {'random_contiguous', 'fixed_contiguous'}:
        raise ValueError(f'Unsupported missing_time_mode: {missing_time_mode}')

    candidate_starts = [ts for ts in unique_times if ts + duration <= unique_times[-1]]
    if not candidate_starts:
        if logger is not None:
            logger.warning(f'No valid contiguous window for missing_time_hours={missing_time_hours}; keep original train set.')
        return data, None

    if missing_time_mode == 'fixed_contiguous':
        start_time = candidate_starts[0]
    else:
        rng = np.random.default_rng(seed)
        start_time = candidate_starts[int(rng.integers(0, len(candidate_starts)))]
    end_time = start_time + duration

    keep_mask = np.array([(ts < start_time) or (ts >= end_time) for ts in train_times], dtype=bool)
    removed_count = int((~keep_mask).sum())
    if logger is not None:
        logger.info(
            f'Apply contiguous train-time drop: start={start_time}, end={end_time}, '
            f'removed={removed_count}/{len(train_times)} samples'
        )

    data['x_train'] = data['x_train'][keep_mask]
    data['y_train'] = data['y_train'][keep_mask]
    return data, (start_time, end_time, removed_count)


def _drop_calendar_based_train_windows(data, pattern='none', mode='random', seed=0, logger=None):
    if pattern == 'none':
        return data, None

    pattern_to_hours = {
        'daily_1h': 1,
        'weekly_1day': 24,
        'monthly_1week': 168,
    }
    if pattern not in pattern_to_hours:
        raise ValueError(f'Unsupported calendar pattern: {pattern}')
    if mode not in {'random', 'fixed'}:
        raise ValueError(f'Unsupported calendar mode: {mode}')

    drop_hours = pattern_to_hours[pattern]
    train_times = [_sample_end_timestamp(data['x_train'][index]) for index in range(len(data['x_train']))]
    if not train_times:
        return data, None

    grouped_indices = defaultdict(list)
    grouped_times = defaultdict(list)
    for index, timestamp in enumerate(train_times):
        group_key = _calendar_group_key(timestamp, pattern)
        grouped_indices[group_key].append(index)
        grouped_times[group_key].append(timestamp)

    rng = np.random.default_rng(seed)
    keep_mask = np.ones(len(train_times), dtype=bool)
    removed_total = 0
    dropped_windows = []

    for group_key in sorted(grouped_indices.keys()):
        group_indices = grouped_indices[group_key]
        group_times = sorted(set(grouped_times[group_key]))
        if not group_times:
            continue

        duration = timedelta(hours=float(drop_hours))
        candidate_starts = [ts for ts in group_times if ts + duration <= group_times[-1]]
        if not candidate_starts:
            if logger is not None:
                logger.warning(f'No valid {pattern} window for group={group_key}; skip this group.')
            continue

        if mode == 'fixed':
            start_time = candidate_starts[0]
        else:
            start_time = candidate_starts[int(rng.integers(0, len(candidate_starts)))]
        end_time = start_time + duration

        group_removed = 0
        for index in group_indices:
            timestamp = train_times[index]
            if start_time <= timestamp < end_time:
                keep_mask[index] = False
                group_removed += 1

        removed_total += group_removed
        dropped_windows.append((group_key, start_time, end_time, group_removed))

    if logger is not None:
        logger.info(
            f'Apply calendar drop pattern={pattern}, mode={mode}, removed={removed_total}/{len(train_times)} samples'
        )
        for group_key, start_time, end_time, group_removed in dropped_windows:
            logger.info(f'  group={group_key}, start={start_time}, end={end_time}, removed={group_removed}')

    data['x_train'] = data['x_train'][keep_mask]
    data['y_train'] = data['y_train'][keep_mask]
    return data, dropped_windows
    

def get_dataloaders_scaler(dataset_dir, batch_size=16, logger=None, output_len=None, missing_tail_steps=0, missing_tail_node_ratio=0.0, missing_tail_seed=0):
    
    data = {}
    datasets = {}
    dataloaders = {}
    num_samples = 0
    
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        y = cat_data['y'][..., :1]
        y = _slice_output(y, output_len)
        data['y_' + category] = y
        num_samples += data['x_' + category].shape[0]
        
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        datasets[category] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_' + category]), torch.FloatTensor(data['y_' + category]))
    
    # (num_samples, length, num_nodes, dim)
    logger.info(f"Data Length: {num_samples} Node num: {data['x_train'].shape[2]}")
    logger.info(f"Train num: {data['x_train'].shape[0]} Val num: {data['x_val'].shape[0]} Test num: {data['x_test'].shape[0]}")

    if missing_tail_steps > 0:
        num_nodes = data['x_train'].shape[2]
        tail_missing_nodes = None
        if missing_tail_node_ratio > 0:
            tail_missing_nodes = _select_fixed_missing_nodes(num_nodes, missing_tail_node_ratio, seed=missing_tail_seed)
            if tail_missing_nodes.size == 0:
                tail_missing_nodes = None
            if logger is not None:
                logger.info(
                    f'Fixed tail-step missing nodes selected once: ratio={missing_tail_node_ratio}, '
                    f'count={0 if tail_missing_nodes is None else len(tail_missing_nodes)}/{num_nodes}, '
                    f'nodes={[] if tail_missing_nodes is None else tail_missing_nodes.tolist()}'
                )
        elif logger is not None:
            logger.info(f'Apply tail-step missing: steps={missing_tail_steps}, nodes=all')

        _apply_tail_missing(data['x_train'], np.arange(data['x_train'].shape[0]), missing_tail_steps, tail_missing_nodes)
        _apply_tail_missing(data['x_val'], np.arange(data['x_val'].shape[0]), missing_tail_steps, tail_missing_nodes)
        _apply_tail_missing(data['x_test'], np.arange(data['x_test'].shape[0]), missing_tail_steps, tail_missing_nodes)

    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)

    return dataloaders, scaler


# def get_dataloaders_scaler_and_split_task(dataset_dir, batch_size=16, task_per_dir=4, logger=None):
    
#     data = {}
#     scalers = []
#     datasets = {}
#     dataset = {}
#     dataloader = {}
#     num_samples = 0
#     dataloaders = []
#     train_tasks = {}
#     val_tasks = {}
#     test_tasks = {}
#     for i in range(task_per_dir):
#         train_tasks[i] = []
#         val_tasks[i] = []
#         test_tasks[i] = []
#     for category in ['train', 'val', 'test']:
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
#         data['x_' + category] = cat_data['x']
#         data['y_' + category] = cat_data['y'][...,:1]
#         num_samples += data['x_' + category].shape[0]
        
#     #scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
#     # Data format
#     # for category in ['train', 'val', 'test']:
#     #     data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
#         #datasets[category] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_' + category]), torch.FloatTensor(data['y_' + category]))
#     num_nodes = data['x_train'].shape[2]
#     # (num_samples, length, num_nodes, dim)
#     # logger.info(f"Data Length: {num_samples} Node num: {data['x_train'].shape[2]}")
#     # logger.info(f"Train num: {data['x_train'].shape[0]} Val num: {data['x_val'].shape[0]} Test num: {data['x_test'].shape[0]}")
#     for i in range(len(data['x_train'])):  # 根据timestamp划分
#         hour = data['x_train'][i,11,0,8]  # 取(i,11,0,8)作为判断小时的点
#         train_tasks[hour//6].append(i)
#     for i in range(len(data['x_val'])):  # 根据timestamp划分
#         hour = data['x_val'][i,11,0,8]  # 取(i,11,0,8)作为判断小时的点
#         val_tasks[hour//6].append(i)
#     for i in range(len(data['x_test'])):  # 根据timestamp划分
#         hour = data['x_test'][i,11,0,8]  # 取(i,11,0,8)作为判断小时的点
#         test_tasks[hour//6].append(i)
#     for i in range(task_per_dir):
#         # dataset['train'] = datasets['train'][train_tasks[i]]
#         # dataset['val'] = datasets['val'][val_tasks[i]]
#         # dataset['test'] = datasets['test'][test_tasks[i]]
#         scaler = StandardScaler(mean=data['x_train'][train_tasks[i],..., 0].mean(), std=data['x_train'][train_tasks[i],..., 0].std())
#         data['x_train'][train_tasks[i], ..., 0] = scaler.transform(data['x_train'][train_tasks[i],..., 0])
#         data['x_val'][val_tasks[i], ..., 0] = scaler.transform(data['x_val'][val_tasks[i], ..., 0])
#         data['x_test'][test_tasks[i], ..., 0] = scaler.transform(data['x_test'][test_tasks[i], ..., 0])
#         dataset['train'] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_train'][train_tasks[i]]),
#                                                           torch.FloatTensor(data['y_train'][train_tasks[i]]))
#         dataset['val'] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_val'][val_tasks[i]]),
#                                                           torch.FloatTensor(data['y_val'][val_tasks[i]]))
#         dataset['test'] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_test'][test_tasks[i]]),
#                                                           torch.FloatTensor(data['y_test'][test_tasks[i]]))
#     #print(data['x_train'][0,:,0,1])
#         dataloader['train'] = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
#         dataloader['val'] = torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size, shuffle=False)
#         dataloader['test'] = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
#         dataloaders.append(dataloader)
#         scalers.append(scaler)
#     return dataloaders, scalers


def get_dataloaders_scaler_and_split_task(dataset_dir, batch_size=16, task_per_dir=4, logger=None, missing_time_hours=0, missing_time_mode='random_contiguous', missing_time_seed=0, missing_calendar_pattern='none', missing_calendar_mode='random', missing_node_ratio=0.0, output_len=None, missing_tail_steps=0, missing_tail_node_ratio=0.0, missing_tail_seed=0):
    logger.info("=== get_dataloaders_scaler_and_split_task START ===")
    data = {}
    # 1. 读取 train/val/test .npz 文件，并存到 data dict
    for category in ['train', 'val', 'test']:
        file_path = os.path.join(dataset_dir, category + '.npz')
        logger.info(f"Loading {category} data from {file_path} ...")
        cat_data = np.load(file_path)
        
        data['x_' + category] = cat_data['x']              # shape: (num_samples, length, num_nodes, dim)
        y = cat_data['y'][..., :1]     # 只取前 1 个输出特征
        y = _slice_output(y, output_len)
        data['y_' + category] = y

        logger.info(f"{category} data shape: x_{category}={data['x_'+category].shape}, "
              f"y_{category}={data['y_'+category].shape}")

    raw_data = {key: value.copy() for key, value in data.items()}

    raw_data, dropped_window = _drop_calendar_based_train_windows(
        raw_data,
        pattern=missing_calendar_pattern,
        mode=missing_calendar_mode,
        seed=missing_time_seed,
        logger=logger,
    )
    if missing_calendar_pattern == 'none':
        raw_data, dropped_window = _drop_contiguous_train_time_window(
            raw_data,
            missing_time_hours=missing_time_hours,
            missing_time_mode=missing_time_mode,
            seed=missing_time_seed,
            logger=logger,
        )

    data = {key: value.copy() for key, value in raw_data.items()}

    # 2. 打印 num_nodes
    num_nodes = raw_data['x_train'].shape[2]
    logger.info(f"Number of nodes: {num_nodes}")

    missing_nodes = _select_fixed_missing_nodes(num_nodes, missing_node_ratio, seed=missing_time_seed)
    if missing_nodes.size > 0:
        logger.info(
            f'Fixed spatial missing nodes selected once: ratio={missing_node_ratio}, '
            f'count={len(missing_nodes)}/{num_nodes}, nodes={missing_nodes.tolist()}'
        )

    tail_missing_nodes = None
    if missing_tail_steps > 0:
        if missing_tail_node_ratio > 0:
            tail_missing_nodes = _select_fixed_missing_nodes(num_nodes, missing_tail_node_ratio, seed=missing_tail_seed)
            if tail_missing_nodes.size == 0:
                tail_missing_nodes = None
            logger.info(
                f'Fixed tail-step missing nodes selected once: ratio={missing_tail_node_ratio}, '
                f'count={0 if tail_missing_nodes is None else len(tail_missing_nodes)}/{num_nodes}, '
                f'nodes={[] if tail_missing_nodes is None else tail_missing_nodes.tolist()}'
            )
        else:
            logger.info(f'Apply tail-step missing: steps={missing_tail_steps}, nodes=all')

    # 3. 获取 hour 数组
    logger.info("Extracting hours from x_train/x_val/x_test...")
    hour_train = raw_data['x_train'][:, 11, 0, 8].astype(int)  # shape: (num_train_samples,)
    hour_val   = raw_data['x_val'][:,   11, 0, 8].astype(int)  # shape: (num_val_samples,)
    hour_test  = raw_data['x_test'][:,  11, 0, 8].astype(int)  # shape: (num_test_samples,)

    logger.info(f"hour_train shape = {hour_train.shape}, hour_val shape = {hour_val.shape}, hour_test shape = {hour_test.shape}")

    # 4. 计算每个样本所属的分组 ID
    logger.info("Calculating group IDs for each sample...")
    group_train = hour_train // (24 // task_per_dir)  # 每个样本属于哪个组
    group_val   = hour_val   // (24 // task_per_dir)
    group_test  = hour_test  // (24 // task_per_dir)

    # 5. 用 np.where() 找到各组的样本索引
    train_tasks = [np.where(group_train == i)[0] for i in range(task_per_dir)]
    val_tasks   = [np.where(group_val   == i)[0] for i in range(task_per_dir)]
    test_tasks  = [np.where(group_test  == i)[0] for i in range(task_per_dir)]

    for i in range(task_per_dir):
        logger.info(f"Group {i} -> train_samples: {len(train_tasks[i])}, "
              f"val_samples: {len(val_tasks[i])}, test_samples: {len(test_tasks[i])}")

    dataloaders = []
    scalers = []
    cur_means = []
    cur_stds = []
    # 6. 遍历每个分组并创建 DataLoader
    logger.info("Creating DataLoaders for each group/task...")
    for i in range(task_per_dir):
        logger.info(f"=== Now processing group {i} ===")
        sel_train_idx = train_tasks[i]

        sel_val_idx   = val_tasks[i]
        sel_test_idx  = test_tasks[i]

        # 如果该分组训练集是空的，做个提示
        if len(sel_train_idx) == 0:
            msg = f"[Warning] No training samples in group {i}."
            logger.info(msg)
            if logger is not None:
                logger.warning(msg)
            # 如果要跳过这个分组:
            # continue
            # 或者继续往下也行，看你实际需求
            # 这里演示一下继续往下，但需要自己判断是否会影响后续流程

        # 6.1 创建 StandardScaler
        mean_val = raw_data['x_train'][sel_train_idx, ..., 0].mean() if len(sel_train_idx) > 0 else 0
        std_val  = raw_data['x_train'][sel_train_idx, ..., 0].std()  if len(sel_train_idx) > 0 else 1e-8
        feature_dim = raw_data['x_train'].shape[-1]
        cur_mean = np.mean(raw_data['x_train'][sel_train_idx], axis=(0,1,2)) if len(sel_train_idx) > 0 else np.zeros(feature_dim)
        cur_std = np.std(raw_data['x_train'][sel_train_idx], axis=(0,1,2)) if len(sel_train_idx) > 0 else np.zeros(feature_dim) + (1e-8)
        scaler = StandardScaler(mean=mean_val, std=std_val)

        logger.info(f"Scaler for group {i}: mean={scaler.mean}, std={scaler.std}")

        # 6.2 对 train/val/test 数据做 transform
        data['x_train'][sel_train_idx, ..., 0] = scaler.transform(raw_data['x_train'][sel_train_idx, ..., 0])
        data['x_val'][sel_val_idx,     ..., 0] = scaler.transform(raw_data['x_val'][sel_val_idx,     ..., 0])
        data['x_test'][sel_test_idx,   ..., 0] = scaler.transform(raw_data['x_test'][sel_test_idx,   ..., 0])

        if missing_nodes.size > 0:
            time_len = data['x_train'].shape[1]
            feat_len = data['x_train'].shape[3]
            train_time_idx = np.arange(time_len)
            feat_idx = np.arange(feat_len)
            data['x_train'][np.ix_(sel_train_idx, train_time_idx, missing_nodes, feat_idx)] = 0
            data['x_val'][np.ix_(sel_val_idx, train_time_idx, missing_nodes, feat_idx)] = 0
            data['x_test'][np.ix_(sel_test_idx, train_time_idx, missing_nodes, feat_idx)] = 0

        if missing_tail_steps > 0:
            _apply_tail_missing(data['x_train'], sel_train_idx, missing_tail_steps, tail_missing_nodes)
            _apply_tail_missing(data['x_val'], sel_val_idx, missing_tail_steps, tail_missing_nodes)
            _apply_tail_missing(data['x_test'], sel_test_idx, missing_tail_steps, tail_missing_nodes)

        # 6.3 创建 Dataset
        logger.info(f"Constructing TensorDataset for group {i}...")
        dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_train'][sel_train_idx]),
            torch.FloatTensor(data['y_train'][sel_train_idx])
        )
        dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_val'][sel_val_idx]),
            torch.FloatTensor(data['y_val'][sel_val_idx])
        )
        dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_test'][sel_test_idx]),
            torch.FloatTensor(data['y_test'][sel_test_idx])
        )

        # 6.4 创建 DataLoader
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val,   batch_size=batch_size, shuffle=False)
        dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

        dataloader_dict = {
            "train": dataloader_train,
            "val":   dataloader_val,
            "test":  dataloader_test
        }

        dataloaders.append(dataloader_dict)
        scalers.append(scaler)
        cur_means.append(cur_mean)
        cur_stds.append(cur_std)
        logger.info(f"Group {i} DataLoader created: "
              f"train_size={len(dataset_train)}, val_size={len(dataset_val)}, test_size={len(dataset_test)}")

    logger.info("=== get_dataloaders_scaler_and_split_task DONE ===")
    return dataloaders, scalers, cur_means, cur_stds

def get_dataloaders_scaler_and_split_task_few_shot(dataset_dir, batch_size=16, task_per_dir=4, logger=None, few_shot_scale=1, missing_time_hours=0, missing_time_mode='random_contiguous', missing_time_seed=0, missing_calendar_pattern='none', missing_calendar_mode='random', missing_node_ratio=0.0, output_len=None, missing_tail_steps=0, missing_tail_node_ratio=0.0, missing_tail_seed=0):
    logger.info("=== get_dataloaders_scaler_and_split_task START ===")
    data = {}
    # 1. 读取 train/val/test .npz 文件，并存到 data dict
    for category in ['train', 'val', 'test']:
        file_path = os.path.join(dataset_dir, category + '.npz')
        logger.info(f"Loading {category} data from {file_path} ...")
        cat_data = np.load(file_path)
        
        data['x_' + category] = cat_data['x']              # shape: (num_samples, length, num_nodes, dim)
        y = cat_data['y'][..., :1]     # 只取前 1 个输出特征
        y = _slice_output(y, output_len)
        data['y_' + category] = y

        logger.info(f"{category} data shape: x_{category}={data['x_'+category].shape}, "
              f"y_{category}={data['y_'+category].shape}")

    raw_data = {key: value.copy() for key, value in data.items()}

    raw_data, dropped_window = _drop_calendar_based_train_windows(
        raw_data,
        pattern=missing_calendar_pattern,
        mode=missing_calendar_mode,
        seed=missing_time_seed,
        logger=logger,
    )
    if missing_calendar_pattern == 'none':
        raw_data, dropped_window = _drop_contiguous_train_time_window(
            raw_data,
            missing_time_hours=missing_time_hours,
            missing_time_mode=missing_time_mode,
            seed=missing_time_seed,
            logger=logger,
        )

    data = {key: value.copy() for key, value in raw_data.items()}

    # 2. 打印 num_nodes
    num_nodes = raw_data['x_train'].shape[2]
    logger.info(f"Number of nodes: {num_nodes}")

    missing_nodes = _select_fixed_missing_nodes(num_nodes, missing_node_ratio, seed=missing_time_seed)
    if missing_nodes.size > 0:
        logger.info(
            f'Fixed spatial missing nodes selected once: ratio={missing_node_ratio}, '
            f'count={len(missing_nodes)}/{num_nodes}, nodes={missing_nodes.tolist()}'
        )

    tail_missing_nodes = None
    if missing_tail_steps > 0:
        if missing_tail_node_ratio > 0:
            tail_missing_nodes = _select_fixed_missing_nodes(num_nodes, missing_tail_node_ratio, seed=missing_tail_seed)
            if tail_missing_nodes.size == 0:
                tail_missing_nodes = None
            logger.info(
                f'Fixed tail-step missing nodes selected once: ratio={missing_tail_node_ratio}, '
                f'count={0 if tail_missing_nodes is None else len(tail_missing_nodes)}/{num_nodes}, '
                f'nodes={[] if tail_missing_nodes is None else tail_missing_nodes.tolist()}'
            )
        else:
            logger.info(f'Apply tail-step missing: steps={missing_tail_steps}, nodes=all')

    # 3. 获取 hour 数组
    logger.info("Extracting hours from x_train/x_val/x_test...")
    hour_train = raw_data['x_train'][:, 11, 0, 8].astype(int)  # shape: (num_train_samples,)
    hour_val   = raw_data['x_val'][:,   11, 0, 8].astype(int)  # shape: (num_val_samples,)
    hour_test  = raw_data['x_test'][:,  11, 0, 8].astype(int)  # shape: (num_test_samples,)

    logger.info(f"hour_train shape = {hour_train.shape}, hour_val shape = {hour_val.shape}, hour_test shape = {hour_test.shape}")

    # 4. 计算每个样本所属的分组 ID
    logger.info("Calculating group IDs for each sample...")
    group_train = hour_train // (24 // task_per_dir)  # 每个样本属于哪个组
    group_val   = hour_val   // (24 // task_per_dir)
    group_test  = hour_test  // (24 // task_per_dir)

    # 5. 用 np.where() 找到各组的样本索引
    train_tasks = [np.where(group_train == i)[0] for i in range(task_per_dir)]
    val_tasks   = [np.where(group_val   == i)[0] for i in range(task_per_dir)]
    test_tasks  = [np.where(group_test  == i)[0] for i in range(task_per_dir)]

    for i in range(task_per_dir):
        logger.info(f"Group {i} -> train_samples: {len(train_tasks[i])}, "
              f"val_samples: {len(val_tasks[i])}, test_samples: {len(test_tasks[i])}")

    dataloaders = []
    scalers = []
    cur_means = []
    cur_stds = []
    # 6. 遍历每个分组并创建 DataLoader
    logger.info("Creating DataLoaders for each group/task...")
    for i in range(task_per_dir):
        logger.info(f"=== Now processing group {i} ===")
        sel_train_idx = train_tasks[i]
        logger.info(f'sel_train_idx: {sel_train_idx}, {type(sel_train_idx)}')
        few_train_num = int(len(sel_train_idx) * few_shot_scale)
        sel_train_idx = np.random.choice(sel_train_idx, size=few_train_num, replace=False)
        sel_val_idx   = val_tasks[i]
        sel_test_idx  = test_tasks[i]

        # 如果该分组训练集是空的，做个提示
        if len(sel_train_idx) == 0:
            msg = f"[Warning] No training samples in group {i}."
            print(msg)
            if logger is not None:
                logger.warning(msg)
            # 如果要跳过这个分组:
            # continue
            # 或者继续往下也行，看你实际需求
            # 这里演示一下继续往下，但需要自己判断是否会影响后续流程

        # 6.1 创建 StandardScaler
        mean_val = raw_data['x_train'][sel_train_idx, ..., 0].mean() if len(sel_train_idx) > 0 else 0
        std_val  = raw_data['x_train'][sel_train_idx, ..., 0].std()  if len(sel_train_idx) > 0 else 1e-8
        feature_dim = raw_data['x_train'].shape[-1]
        cur_mean = np.mean(raw_data['x_train'][sel_train_idx], axis=(0,1,2)) if len(sel_train_idx) > 0 else np.zeros(feature_dim)
        cur_std = np.std(raw_data['x_train'][sel_train_idx], axis=(0,1,2)) if len(sel_train_idx) > 0 else np.zeros(feature_dim) + (1e-8)
        scaler = StandardScaler(mean=mean_val, std=std_val)

        print(f"Scaler for group {i}: mean={scaler.mean}, std={scaler.std}")

        # 6.2 对 train/val/test 数据做 transform
        data['x_train'][sel_train_idx, ..., 0] = scaler.transform(raw_data['x_train'][sel_train_idx, ..., 0])
        data['x_val'][sel_val_idx,     ..., 0] = scaler.transform(raw_data['x_val'][sel_val_idx,     ..., 0])
        data['x_test'][sel_test_idx,   ..., 0] = scaler.transform(raw_data['x_test'][sel_test_idx,   ..., 0])

        if missing_nodes.size > 0:
            time_len = data['x_train'].shape[1]
            feat_len = data['x_train'].shape[3]
            train_time_idx = np.arange(time_len)
            feat_idx = np.arange(feat_len)
            data['x_train'][np.ix_(sel_train_idx, train_time_idx, missing_nodes, feat_idx)] = 0
            data['x_val'][np.ix_(sel_val_idx, train_time_idx, missing_nodes, feat_idx)] = 0
            data['x_test'][np.ix_(sel_test_idx, train_time_idx, missing_nodes, feat_idx)] = 0

        if missing_tail_steps > 0:
            _apply_tail_missing(data['x_train'], sel_train_idx, missing_tail_steps, tail_missing_nodes)
            _apply_tail_missing(data['x_val'], sel_val_idx, missing_tail_steps, tail_missing_nodes)
            _apply_tail_missing(data['x_test'], sel_test_idx, missing_tail_steps, tail_missing_nodes)

        # 6.3 创建 Dataset
        print(f"Constructing TensorDataset for group {i}...")
        dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_train'][sel_train_idx]),
            torch.FloatTensor(data['y_train'][sel_train_idx])
        )
        dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_val'][sel_val_idx]),
            torch.FloatTensor(data['y_val'][sel_val_idx])
        )
        dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(data['x_test'][sel_test_idx]),
            torch.FloatTensor(data['y_test'][sel_test_idx])
        )

        # 6.4 创建 DataLoader
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val,   batch_size=batch_size, shuffle=False)
        dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

        dataloader_dict = {
            "train": dataloader_train,
            "val":   dataloader_val,
            "test":  dataloader_test
        }

        dataloaders.append(dataloader_dict)
        scalers.append(scaler)
        cur_means.append(cur_mean)
        cur_stds.append(cur_std)
        print(f"Group {i} DataLoader created: "
              f"train_size={len(dataset_train)}, val_size={len(dataset_val)}, test_size={len(dataset_test)}")

    print("=== get_dataloaders_scaler_and_split_task DONE ===")
    return dataloaders, scalers, cur_means, cur_stds