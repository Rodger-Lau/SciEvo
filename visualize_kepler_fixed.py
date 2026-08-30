
"""
visualize_kepler.py
基于 keplergl 0.2.1 的 PM2.5 预测结果地图可视化模块
适配 main_Knowair_2d.py 的输出格式

修复记录：
  - 修复 KeyError: 'latitude' → 自动检测 CSV 列名
  - 修复 sample → station 映射逻辑
  - 增加诊断日志，便于排查问题
"""

import os
import numpy as np
import pandas as pd
from keplergl import KeplerGl
import json
import warnings
warnings.filterwarnings('ignore')


# ======================== AQI 颜色映射配置 ======================== #
PM25_COLOR_MAP = {
    'range': [0, 35, 75, 115, 150, 250, 500],
    'colors': ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#99004c', '#7e0023'],
    'labels': ['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染']
}


def pm25_to_aqi_level(value):
    """将 PM2.5 值转换为 AQI 等级标签"""
    if value <= 35:
        return '优'
    elif value <= 75:
        return '良'
    elif value <= 115:
        return '轻度污染'
    elif value <= 150:
        return '中度污染'
    elif value <= 250:
        return '重度污染'
    else:
        return '严重污染'


# ======================== CSV 列名自动检测 ======================== #
# 已知的各种列名变体（小写化后匹配）
LAT_CANDIDATES = ['latitude', 'lat', 'lat_', 'y', '纬度', 'lat_deg', 'station_lat']
LON_CANDIDATES = ['longitude', 'lon', 'lng', 'long', 'x', '经度', 'lon_deg', 'station_lon']
NAME_CANDIDATES = ['name', 'station_name', 'city_name', 'city', 'station', '站点名称',
                   '城市', '城市名称', 'location', 'site_name', 'site']
ID_CANDIDATES = ['id', 'station_id', 'index', 'station_idx', '编号', '站点编号',
                 'fid', 'oid', 'no', 'seq']


def _find_column(columns, candidates, fallback_idx=None):
    """
    在 columns 列表中查找匹配候选名的列（不区分大小写，去除空格和下划线）

    参数:
        columns: 实际列名列表
        candidates: 候选列名列表
        fallback_idx: 如果找不到，使用的回退索引
    返回:
        匹配到的列名，或 None
    """
    # 标准化函数：小写 + 去空格/下划线
    def normalize(s):
        return s.lower().strip().replace(' ', '').replace('_', '').replace('-', '')

    norm_candidates = [normalize(c) for c in candidates]
    norm_columns = {normalize(c): c for c in columns}

    for nc in norm_candidates:
        if nc in norm_columns:
            return norm_columns[nc]

    # 模糊匹配：候选名包含在列名中，或列名包含在候选名中
    for nc in norm_candidates:
        for norm_col, orig_col in norm_columns.items():
            if nc in norm_col or norm_col in nc:
                return orig_col

    # 回退
    if fallback_idx is not None and fallback_idx < len(columns):
        return columns[fallback_idx]

    return None


def diagnose_csv(csv_path):
    """
    诊断 CSV 文件，打印列名和前几行数据，帮助调试

    返回:
        list: 列名列表
    """
    df = pd.read_csv(csv_path)
    print(f"  CSV 文件: {csv_path}")
    print(f"  列名: {list(df.columns)}")
    print(f"  行数: {len(df)}")
    print(f"  前3行预览:")
    print(df.head(3).to_string(index=False))

    # 尝试自动检测
    lat_col = _find_column(df.columns, LAT_CANDIDATES, fallback_idx=0)
    lon_col = _find_column(df.columns, LON_CANDIDATES, fallback_idx=1)
    name_col = _find_column(df.columns, NAME_CANDIDATES)
    id_col = _find_column(df.columns, ID_CANDIDATES, fallback_idx=0)
    print(f"  自动检测结果:")
    print(f"    纬度列 → '{lat_col}'")
    print(f"    经度列 → '{lon_col}'")
    print(f"    名称列 → '{name_col}'")
    print(f"    编号列 → '{id_col}'")

    return list(df.columns)


def load_station_coords(graph_obj=None, coords_file=None):
    """
    加载站点经纬度坐标（支持多种格式和列名）

    参数:
        graph_obj: Graph 对象（优先使用）
        coords_file: 站点坐标文件路径（.csv / .json / .npy）

    返回:
        dict: {station_idx: {'lat': float, 'lon': float, 'name': str}}
    """
    stations = {}

    # ---- 优先从 Graph 对象提取 ----
    if graph_obj is not None:
        if hasattr(graph_obj, 'node_coords'):
            coords = graph_obj.node_coords
            if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] >= 2:
                for idx in range(coords.shape[0]):
                    stations[idx] = {
                        'lat': float(coords[idx, 1]),
                        'lon': float(coords[idx, 0]),
                        'name': f'Station_{idx}'
                    }
                print(f"✅ 从 graph.node_coords 加载了 {len(stations)} 个站点坐标")

        elif hasattr(graph_obj, 'pos'):
            coords = graph_obj.pos
            if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] >= 2:
                for idx in range(coords.shape[0]):
                    stations[idx] = {
                        'lat': float(coords[idx, 1]),
                        'lon': float(coords[idx, 0]),
                        'name': f'Station_{idx}'
                    }
                print(f"✅ 从 graph.pos 加载了 {len(stations)} 个站点坐标")

        elif hasattr(graph_obj, 'node_lat') and hasattr(graph_obj, 'node_lon'):
            lats = graph_obj.node_lat
            lons = graph_obj.node_lon
            for idx in range(len(lats)):
                stations[idx] = {
                    'lat': float(lats[idx]),
                    'lon': float(lons[idx]),
                    'name': f'Station_{idx}'
                }
            print(f"✅ 从 graph.node_lat/node_lon 加载了 {len(stations)} 个站点坐标")

        # 尝试从 node_attr 提取
        elif hasattr(graph_obj, 'node_attr') or hasattr(graph_obj, 'x'):
            node_attr = getattr(graph_obj, 'node_attr', None) or getattr(graph_obj, 'x', None)
            if isinstance(node_attr, np.ndarray) and node_attr.ndim == 2 and node_attr.shape[1] >= 2:
                for idx in range(node_attr.shape[0]):
                    stations[idx] = {
                        'lat': float(node_attr[idx, 1]),
                        'lon': float(node_attr[idx, 0]),
                        'name': f'Station_{idx}'
                    }
                print(f"✅ 从 graph node_attr 加载了 {len(stations)} 个站点坐标")

    if len(stations) > 0:
        return stations

    # ---- 从文件加载 ----
    if coords_file is not None and os.path.exists(coords_file):
        print(f"正在从文件加载站点坐标: {coords_file}")

        if coords_file.endswith('.csv'):
            stations = _load_from_csv(coords_file)
        elif coords_file.endswith('.json'):
            stations = _load_from_json(coords_file)
        elif coords_file.endswith('.npy'):
            stations = _load_from_npy(coords_file)
        else:
            print(f"⚠️ 不支持的文件格式: {coords_file}")

    if len(stations) == 0:
        print("❌ 未能加载任何站点坐标！")
        print("   请检查文件路径和列名是否正确")
    else:
        # 验证坐标合理性（中国范围：纬度 18~54, 经度 73~135）
        lats = [v['lat'] for v in stations.values()]
        lons = [v['lon'] for v in stations.values()]
        lat_in_china = all(18 <= la <= 54 for la in lats)
        lon_in_china = all(73 <= lo <= 135 for lo in lons)

        if not lat_in_china or not lon_in_china:
            print(f"⚠️ 坐标可能不在中国范围内:")
            print(f"   纬度范围: [{min(lats):.2f}, {max(lats):.2f}] (期望 18~54)")
            print(f"   经度范围: [{min(lons):.2f}, {max(lons):.2f}] (期望 73~135)")
            print(f"   可能经纬度列顺序反了，尝试交换...")

            # 尝试交换经纬度
            if all(18 <= lo <= 54 for lo in lons) and all(73 <= la <= 135 for la in lats):
                print(f"   ✅ 交换经纬度后范围正确，自动修正")
                for k in stations:
                    stations[k]['lat'], stations[k]['lon'] = stations[k]['lon'], stations[k]['lat']
            elif not lat_in_china and not lon_in_china:
                print(f"   ❌ 交换后仍不在范围内，请手动检查坐标数据")
        else:
            print(f"✅ 坐标范围验证通过 (中国区域)")

        print(f"✅ 共加载 {len(stations)} 个站点坐标")
        # 打印前3个
        for idx in list(stations.keys())[:3]:
            info = stations[idx]
            print(f"   站点 {idx}: lat={info['lat']:.4f}, lon={info['lon']:.4f}, name={info['name']}")

    return stations


def _load_from_csv(csv_path):
    """从 CSV 文件加载站点坐标，自动检测列名"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return {}

    # 诊断
    print(f"  CSV 列名: {list(df.columns)}")
    print(f"  CSV 行数: {len(df)}")
    print(f"  前2行:")
    print(df.head(2).to_string(index=False))

    # 自动检测列名
    lat_col = _find_column(df.columns, LAT_CANDIDATES, fallback_idx=0)
    lon_col = _find_column(df.columns, LON_CANDIDATES, fallback_idx=1)
    name_col = _find_column(df.columns, NAME_CANDIDATES)
    id_col = _find_column(df.columns, ID_CANDIDATES, fallback_idx=0)

    print(f"  检测结果 → 纬度列: '{lat_col}', 经度列: '{lon_col}', 名称列: '{name_col}', 编号列: '{id_col}'")

    if lat_col is None or lon_col is None:
        print(f"❌ 无法自动检测经纬度列！")
        print(f"   请手动指定列名，或在 CSV 中使用以下列名之一:")
        print(f"   纬度: {LAT_CANDIDATES}")
        print(f"   经度: {LON_CANDIDATES}")
        # 最后尝试：取前两列作为 lat, lon
        if len(df.columns) >= 2:
            lat_col = df.columns[0]
            lon_col = df.columns[1]
            print(f"   尝试使用前两列: '{lat_col}' 作为纬度, '{lon_col}' 作为经度")
        else:
            return {}

    stations = {}
    for idx, row in df.iterrows():
        try:
            lat_val = float(row[lat_col])
            lon_val = float(row[lon_col])
            name_val = str(row[name_col]) if name_col and pd.notna(row[name_col]) else f'Station_{idx}'
            stations[idx] = {
                'lat': lat_val,
                'lon': lon_val,
                'name': name_val
            }
        except (ValueError, TypeError) as e:
            print(f"  ⚠️ 跳过第 {idx} 行（坐标转换失败）: {e}")
            continue

    return stations


def _load_from_json(json_path):
    """从 JSON 文件加载站点坐标"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return {}

    stations = {}
    if isinstance(data, dict):
        for key, val in data.items():
            idx = int(key) if key.isdigit() else key
            if isinstance(val, dict):
                lat = val.get('lat', val.get('latitude', val.get('y', 0)))
                lon = val.get('lon', val.get('longitude', val.get('lng', val.get('x', 0))))
                name = val.get('name', val.get('station_name', f'Station_{idx}'))
                stations[idx] = {'lat': float(lat), 'lon': float(lon), 'name': str(name)}
            elif isinstance(val, (list, tuple)) and len(val) >= 2:
                stations[idx] = {'lat': float(val[1]), 'lon': float(val[0]), 'name': f'Station_{idx}'}
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                lat = item.get('lat', item.get('latitude', item.get('y', 0)))
                lon = item.get('lon', item.get('longitude', item.get('lng', item.get('x', 0))))
                name = item.get('name', f'Station_{idx}')
                stations[idx] = {'lat': float(lat), 'lon': float(lon), 'name': str(name)}

    return stations


def _load_from_npy(npy_path):
    """从 NPY 文件加载站点坐标"""
    try:
        data = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 读取 NPY 失败: {e}")
        return {}

    stations = {}

    if isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] >= 2:
            # 假设是 (N, 2) 的 [lon, lat] 或 [lat, lon] 数组
            for idx in range(data.shape[0]):
                stations[idx] = {
                    'lat': float(data[idx, 1]),
                    'lon': float(data[idx, 0]),
                    'name': f'Station_{idx}'
                }
        elif data.ndim == 0 and hasattr(data.item(), 'items'):
            # 字典格式
            d = data.item()
            for key, val in d.items():
                if isinstance(val, dict):
                    lat = val.get('lat', val.get('latitude', 0))
                    lon = val.get('lon', val.get('longitude', 0))
                    name = val.get('name', f'Station_{key}')
                    stations[key] = {'lat': float(lat), 'lon': float(lon), 'name': str(name)}

    return stations


def save_station_coords(graph_obj, save_path):
    """
    保存站点坐标到文件，供可视化时使用

    参数:
        graph_obj: Graph 对象
        save_path: 保存路径（支持 .npy, .json, .csv）
    """
    # 优先用 load_station_coords 提取
    stations = load_station_coords(graph_obj=graph_obj)

    # 如果 Graph 对象没提取到，尝试从 KnowAir.npy 提取
    if len(stations) == 0:
        stations = _extract_from_knowair_npy()

    if len(stations) == 0:
        print("❌ 无法从 Graph 或 KnowAir.npy 提取站点坐标")
        print("   请手动创建站点坐标文件（CSV 格式，至少包含经纬度两列）")
        return stations

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    if save_path.endswith('.npy'):
        np.save(save_path, stations)
    elif save_path.endswith('.json'):
        with open(save_path, 'w') as f:
            json.dump(stations, f, indent=2, ensure_ascii=False)
    elif save_path.endswith('.csv'):
        rows = []
        for idx, info in stations.items():
            rows.append({
                'latitude': info['lat'],
                'longitude': info['lon'],
                'name': info.get('name', f'Station_{idx}')
            })
        pd.DataFrame(rows).to_csv(save_path, index=False)

    print(f"✅ 站点坐标已保存到: {save_path} (共 {len(stations)} 个站点)")
    return stations


def _extract_from_knowair_npy():
    """从 KnowAir.npy 文件中提取站点坐标（备选方案）"""
    try:
        from util import config
        knowair_fp = config['filepath']['GPU-Server']['knowair_fp']
        if not os.path.exists(knowair_fp):
            print(f"  ⚠️ KnowAir.npy 不存在: {knowair_fp}")
            return {}

        data = np.load(knowair_fp, allow_pickle=True)
        print(f"  KnowAir.npy 形状: {data.shape if hasattr(data, 'shape') else type(data)}")

        if isinstance(data, np.ndarray) and data.ndim == 3:
            num_stations = data.shape[1]
            print(f"  检测到 {num_stations} 个站点")
            # KnowAir.npy 不包含坐标信息
            return {}

        if hasattr(data, 'item'):
            meta = data.item()
            if isinstance(meta, dict):
                for key in ['city_coords', 'station_coords', 'coords', 'lat_lon']:
                    if key in meta:
                        coords = meta[key]
                        stations = {}
                        for idx, coord in enumerate(coords):
                            stations[idx] = {
                                'lat': float(coord[1]) if len(coord) >= 2 else 0,
                                'lon': float(coord[0]) if len(coord) >= 2 else 0,
                                'name': f'Station_{idx}'
                            }
                        return stations
                # 尝试从 city_names 构建
                if 'city_names' in meta:
                    print(f"  找到 city_names 但无坐标信息，需要外部提供坐标")
                    return {}
    except Exception as e:
        print(f"  ❌ 从 KnowAir.npy 提取坐标失败: {e}")

    return {}


# ======================== 预测结果转换 ======================== #
def predictions_to_dataframe(predict_npy, label_npy, time_npy,
                              station_coords, pm25_mean, pm25_std,
                              hist_len=1, pred_len=24,
                              domain_info=None):
    """
    将预测结果的 npy 文件转换为 keplergl 所需的 DataFrame

    参数:
        predict_npy: 预测值数组，形状 (N, seq_len, 1, 1) 或 (N, seq_len, 1)
        label_npy: 真实值数组
        time_npy: 时间数组
        station_coords: 站点坐标字典 {idx: {'lat', 'lon', 'name'}}
        pm25_mean: PM2.5 均值（用于反归一化）
        pm25_std: PM2.5 标准差（用于反归一化）
        hist_len: 历史窗口长度
        pred_len: 预测长度
        domain_info: 域信息字典

    返回:
        pd.DataFrame
    """
    # 反归一化
    pred_val = predict_npy * pm25_std + pm25_mean
    label_val = label_npy * pm25_std + pm25_mean
    pred_val = np.maximum(pred_val, 0)
    label_val = np.maximum(label_val, 0)

    print(f"  pred_val shape: {pred_val.shape}")
    print(f"  label_val shape: {label_val.shape}")

    N = pred_val.shape[0]
    num_stations = len(station_coords)

    print(f"  N (样本数): {N}, num_stations: {num_stations}")

    if num_stations == 0:
        print("❌ 站点坐标为空，无法生成可视化数据")
        return pd.DataFrame()

    # 确定 seq_len 和维度
    if pred_val.ndim == 4:
        seq_len = pred_val.shape[1]
        city_dim = pred_val.shape[2]  # 通常是 1
    elif pred_val.ndim == 3:
        seq_len = pred_val.shape[1]
        city_dim = 1
    else:
        print(f"❌ 不支持的预测值维度: {pred_val.ndim}")
        return pd.DataFrame()

    # 判断数据类型：每个样本是单个站点还是全部站点
    # 情况A: city_dim == 1 → 每个样本是1个站点，N 个样本可能对应不同站点
    # 情况B: city_dim > 1 → 每个样本包含多个站点数据
    # 情况C: city_dim == 1 但 N == num_stations → 一批数据对应所有站点

    rows = []

    if city_dim == 1:
        # 每个样本对应一个站点
        # 映射策略：假设 N 个样本循环对应 num_stations 个站点
        # 例如 N=184, num_stations=184 → 1:1 映射
        # 例如 N=368, num_stations=184 → 每个站点2个样本

        for n in range(N):
            station_idx = n % num_stations

            if station_idx not in station_coords:
                continue

            coord = station_coords[station_idx]

            for t in range(seq_len):
                if t < hist_len:
                    continue

                pred_hour = t - hist_len

                # 提取时间戳
                timestamp = _extract_timestamp(time_npy, n, pred_hour)

                # 提取标量值
                if pred_val.ndim == 4:
                    p_val = float(pred_val[n, t, 0, 0])
                    l_val = float(label_val[n, t, 0, 0])
                elif pred_val.ndim == 3:
                    p_val = float(pred_val[n, t, 0])
                    l_val = float(label_val[n, t, 0])
                else:
                    continue

                row = {
                    'latitude': coord['lat'],
                    'longitude': coord['lon'],
                    'station_name': coord.get('name', f'Station_{station_idx}'),
                    'station_idx': station_idx,
                    'timestamp': str(timestamp),
                    'pred_hour': pred_hour,
                    'pm25_pred': round(p_val, 2),
                    'pm25_true': round(l_val, 2),
                    'pm25_error': round(abs(p_val - l_val), 2),
                    'pm25_rel_error': round(abs(p_val - l_val) / (l_val + 1e-8) * 100, 2),
                    'aqi_level_pred': pm25_to_aqi_level(p_val),
                    'aqi_level_true': pm25_to_aqi_level(l_val),
                }

                if domain_info is not None:
                    row['domain_channel'] = domain_info.get('channel', -1)
                    row['domain_month'] = domain_info.get('month', -1)
                    row['domain_name'] = domain_info.get('name', '')

                rows.append(row)

    elif city_dim > 1 and city_dim == num_stations:
        # 每个样本包含所有站点数据
        for n in range(N):
            for s in range(city_dim):
                if s not in station_coords:
                    continue
                coord = station_coords[s]

                for t in range(seq_len):
                    if t < hist_len:
                        continue
                    pred_hour = t - hist_len
                    timestamp = _extract_timestamp(time_npy, n, pred_hour)

                    if pred_val.ndim == 4:
                        p_val = float(pred_val[n, t, s, 0])
                        l_val = float(label_val[n, t, s, 0])
                    else:
                        continue

                    row = {
                        'latitude': coord['lat'],
                        'longitude': coord['lon'],
                        'station_name': coord.get('name', f'Station_{s}'),
                        'station_idx': s,
                        'timestamp': str(timestamp),
                        'pred_hour': pred_hour,
                        'pm25_pred': round(p_val, 2),
                        'pm25_true': round(l_val, 2),
                        'pm25_error': round(abs(p_val - l_val), 2),
                        'pm25_rel_error': round(abs(p_val - l_val) / (l_val + 1e-8) * 100, 2),
                        'aqi_level_pred': pm25_to_aqi_level(p_val),
                        'aqi_level_true': pm25_to_aqi_level(l_val),
                    }
                    if domain_info is not None:
                        row['domain_channel'] = domain_info.get('channel', -1)
                        row['domain_month'] = domain_info.get('month', -1)
                        row['domain_name'] = domain_info.get('name', '')
                    rows.append(row)

    elif city_dim > 1 and city_dim != num_stations:
        # city_dim 和 num_stations 不匹配，尝试按 city_dim 分配
        print(f"  ⚠️ city_dim={city_dim} != num_stations={num_stations}")
        print(f"     尝试按 city_dim 数量分配站点...")
        for n in range(N):
            for s in range(min(city_dim, num_stations)):
                if s not in station_coords:
                    continue
                coord = station_coords[s]
                for t in range(seq_len):
                    if t < hist_len:
                        continue
                    pred_hour = t - hist_len
                    timestamp = _extract_timestamp(time_npy, n, pred_hour)
                    if pred_val.ndim == 4:
                        p_val = float(pred_val[n, t, s, 0])
                        l_val = float(label_val[n, t, s, 0])
                    else:
                        continue
                    row = {
                        'latitude': coord['lat'],
                        'longitude': coord['lon'],
                        'station_name': coord.get('name', f'Station_{s}'),
                        'station_idx': s,
                        'timestamp': str(timestamp),
                        'pred_hour': pred_hour,
                        'pm25_pred': round(p_val, 2),
                        'pm25_true': round(l_val, 2),
                        'pm25_error': round(abs(p_val - l_val), 2),
                        'pm25_rel_error': round(abs(p_val - l_val) / (l_val + 1e-8) * 100, 2),
                        'aqi_level_pred': pm25_to_aqi_level(p_val),
                        'aqi_level_true': pm25_to_aqi_level(l_val),
                    }
                    if domain_info is not None:
                        row['domain_channel'] = domain_info.get('channel', -1)
                        row['domain_month'] = domain_info.get('month', -1)
                        row['domain_name'] = domain_info.get('name', '')
                    rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  ✅ 生成 DataFrame: {len(df)} 行, {len(df.columns)} 列")
    if len(df) > 0:
        print(f"  列名: {list(df.columns)}")
        print(f"  纬度范围: [{df['latitude'].min():.2f}, {df['latitude'].max():.2f}]")
        print(f"  经度范围: [{df['longitude'].min():.2f}, {df['longitude'].max():.2f}]")

    return df


def _extract_timestamp(time_npy, sample_idx, hour_offset):
    """
    从 time_npy 中提取时间戳，兼容多种格式

    参数:
        time_npy: 时间数组（可能为 None、numpy 数组、或 Unix 时间戳）
        sample_idx: 样本索引
        hour_offset: 小时偏移量

    返回:
        str: 时间戳字符串
    """
    if time_npy is None:
        return f"t+{hour_offset}h"

    try:
        if isinstance(time_npy, np.ndarray):
            if time_npy.ndim == 0:
                # 标量
                t_val = time_npy.item()
            elif time_npy.ndim >= 1 and sample_idx < len(time_npy):
                t_val = time_npy[sample_idx]
                if isinstance(t_val, np.ndarray) and t_val.ndim > 0:
                    t_val = t_val.flat[0]
            else:
                return f"t+{hour_offset}h"

            # 尝试解析为时间戳
            t_val = int(t_val) if np.issubdtype(type(t_val), np.integer) else t_val

            # 如果是 Unix 时间戳（秒或毫秒）
            if isinstance(t_val, (int, np.integer)):
                if t_val > 1e12:  # 毫秒级
                    t_val = t_val / 1000
                try:
                    ts = pd.Timestamp.fromtimestamp(t_val) + pd.Timedelta(hours=hour_offset)
                    return ts.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass

            # 尝试直接构造 Timestamp
            try:
                ts = pd.Timestamp(t_val) + pd.Timedelta(hours=hour_offset)
                return ts.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return str(t_val)

    except Exception:
        pass

    return f"t+{hour_offset}h"


# ======================== Kepler.gl 地图创建 ======================== #
def create_kepler_map(df_pred, df_error=None, map_height=600):
    """创建 kepler.gl 地图"""
    map_1 = KeplerGl(height=map_height)
    map_1.add_data(data=df_pred.copy(), name='PM2.5 Predictions')
    if df_error is not None:
        map_1.add_data(data=df_error.copy(), name='PM2.5 Error Analysis')
    return map_1


def get_kepler_config():
    """返回 kepler.gl 的配置字典"""
    config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [
                    {
                        "dataId": "PM2.5 Predictions",
                        "id": "time_filter",
                        "name": "timestamp",
                        "type": "timeRange",
                        "value": [],
                        "enlarged": True,
                        "plotType": "histogram",
                        "animationWindow": "incremental",
                        "speed": 1
                    },
                    {
                        "dataId": "PM2.5 Predictions",
                        "id": "hour_filter",
                        "name": "pred_hour",
                        "type": "range",
                        "value": [0, 23],
                        "enlarged": False
                    }
                ],
                "layers": [
                    {
                        "id": "pm25_pred_layer",
                        "type": "point",
                        "config": {
                            "dataId": "PM2.5 Predictions",
                            "label": "PM2.5 预测值",
                            "color": [255, 0, 0],
                            "columns": {"lat": "latitude", "lng": "longitude", "altitude": None},
                            "isVisible": True,
                            "visConfig": {
                                "radius": 15,
                                "fixedRadius": False,
                                "opacity": 0.8,
                                "outline": False,
                                "thickness": 2,
                                "colorRange": {
                                    "name": "Custom PM2.5",
                                    "type": "custom",
                                    "category": "Custom",
                                    "colors": PM25_COLOR_MAP['colors']
                                },
                                "radiusRange": [5, 30],
                                "filled": True
                            },
                        },
                        "visualChannels": {
                            "colorField": {"name": "pm25_pred", "type": "real"},
                            "colorScale": "quantile",
                            "sizeField": {"name": "pm25_pred", "type": "real"},
                            "sizeScale": "linear"
                        }
                    },
                    {
                        "id": "pm25_error_layer",
                        "type": "point",
                        "config": {
                            "dataId": "PM2.5 Predictions",
                            "label": "PM2.5 预测误差",
                            "color": [0, 0, 255],
                            "columns": {"lat": "latitude", "lng": "longitude", "altitude": None},
                            "isVisible": False,
                            "visConfig": {
                                "radius": 12,
                                "fixedRadius": False,
                                "opacity": 0.7,
                                "outline": False,
                                "colorRange": {
                                    "name": "Error Heat",
                                    "type": "sequential",
                                    "category": "Uber",
                                    "colors": ["#ffffcc", "#ffeda0", "#fed976", "#feb24c",
                                               "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"]
                                },
                                "radiusRange": [3, 25],
                                "filled": True
                            },
                        },
                        "visualChannels": {
                            "colorField": {"name": "pm25_error", "type": "real"},
                            "colorScale": "quantile",
                            "sizeField": {"name": "pm25_error", "type": "real"},
                            "sizeScale": "linear"
                        }
                    },
                    {
                        "id": "pm25_heatmap_layer",
                        "type": "heatmap",
                        "config": {
                            "dataId": "PM2.5 Predictions",
                            "label": "PM2.5 浓度热力图",
                            "color": [255, 150, 0],
                            "columns": {"lat": "latitude", "lng": "longitude"},
                            "isVisible": False,
                            "visConfig": {
                                "opacity": 0.8,
                                "colorRange": {
                                    "name": "Custom PM2.5",
                                    "type": "custom",
                                    "category": "Custom",
                                    "colors": PM25_COLOR_MAP['colors']
                                },
                                "radius": 30,
                            },
                        },
                        "visualChannels": {
                            "weightField": {"name": "pm25_pred", "type": "real"}
                        }
                    }
                ],
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            "PM2.5 Predictions": [
                                {"name": "station_name", "format": None},
                                {"name": "timestamp", "format": None},
                                {"name": "pred_hour", "format": None},
                                {"name": "pm25_pred", "format": None},
                                {"name": "pm25_true", "format": None},
                                {"name": "pm25_error", "format": None},
                                {"name": "aqi_level_pred", "format": None},
                                {"name": "aqi_level_true", "format": None},
                            ]
                        },
                        "enabled": True
                    },
                    "brush": {"size": 0.5, "enabled": False},
                    "geocoder": {"enabled": True},
                    "coordinate": {"enabled": True}
                },
                "layerBlending": "additive",
                "splitMaps": []
            },
            "mapState": {
                "bearing": 0,
                "dragRotate": False,
                "latitude": 35.0,
                "longitude": 110.0,
                "pitch": 0,
                "zoom": 4,
                "isSplit": False
            },
            "mapStyle": {
                "styleType": "light",
                "topLayerGroups": {},
                "visibleLayerGroups": {
                    "label": True, "road": True, "border": False,
                    "building": True, "water": True, "land": True, "3d building": False
                },
                "threeDBuildingColor": [9.665458101439363, 17.1830549116151, 31.1442869142466],
                "mapStyles": {}
            }
        }
    }
    return config


def generate_summary(df, output_dir):
    """生成预测结果的统计摘要"""
    if len(df) == 0:
        print("⚠️ DataFrame 为空，跳过统计摘要生成")
        return {}

    summary = {
        'total_records': len(df),
        'num_stations': int(df['station_name'].nunique()) if 'station_name' in df.columns else 0,
        'pm25_pred_stats': {
            'mean': float(df['pm25_pred'].mean()),
            'std': float(df['pm25_pred'].std()),
            'min': float(df['pm25_pred'].min()),
            'max': float(df['pm25_pred'].max()),
        },
        'pm25_true_stats': {
            'mean': float(df['pm25_true'].mean()),
            'std': float(df['pm25_true'].std()),
            'min': float(df['pm25_true'].min()),
            'max': float(df['pm25_true'].max()),
        },
        'error_stats': {
            'mae_mean': float(df['pm25_error'].mean()),
            'mae_std': float(df['pm25_error'].std()),
            'mre_mean': float(df['pm25_rel_error'].mean()),
        },
        'aqi_distribution_pred': df['aqi_level_pred'].value_counts().to_dict(),
        'aqi_distribution_true': df['aqi_level_true'].value_counts().to_dict(),
    }

    summary_path = os.path.join(output_dir, 'visualization_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"✅ 统计摘要已保存到: {summary_path}")
    return summary


# ======================== 主可视化函数 ======================== #
def visualize_predictions_kepler(result_dir, station_coords_file,
                                  pm25_mean, pm25_std,
                                  hist_len=1, pred_len=24,
                                  output_dir=None,
                                  save_html=True,
                                  save_csv=True):
    """
    主可视化函数：读取保存的预测结果，生成 kepler.gl 地图

    参数:
        result_dir: 结果目录（包含 predict.npy, label.npy, time.npy）
        station_coords_file: 站点坐标文件路径
        pm25_mean: PM2.5 均值
        pm25_std: PM2.5 标准差
        hist_len: 历史窗口长度
        pred_len: 预测长度
        output_dir: 输出目录
        save_html: 是否保存 HTML 地图
        save_csv: 是否保存 CSV 数据

    返回:
        KeplerGl 地图对象
    """
    print(f"\n{'='*60}")
    print(f"开始生成 Kepler.gl 可视化")
    print(f"结果目录: {result_dir}")
    print(f"站点坐标文件: {station_coords_file}")
    print(f"{'='*60}")

    # ---- 加载站点坐标 ----
    print("\n[1/4] 加载站点坐标...")
    station_coords = load_station_coords(coords_file=station_coords_file)

    if len(station_coords) == 0:
        print("❌ 站点坐标加载失败！")
        print("   请检查站点坐标文件是否存在且格式正确")
        print("   如果是 CSV 文件，请确保包含经纬度列")
        print("   可使用 diagnose_csv() 函数诊断 CSV 文件")
        return None

    # ---- 加载预测结果 ----
    print("\n[2/4] 加载预测结果...")
    predict_path = os.path.join(result_dir, 'predict.npy')
    label_path = os.path.join(result_dir, 'label.npy')
    time_path = os.path.join(result_dir, 'time.npy')

    if not os.path.exists(predict_path):
        print(f"❌ 未找到预测结果文件: {predict_path}")
        return None

    predict_npy = np.load(predict_path)
    label_npy = np.load(label_path)
    time_npy = np.load(time_path) if os.path.exists(time_path) else None

    print(f"  predict.npy shape: {predict_npy.shape}")
    print(f"  label.npy shape: {label_npy.shape}")
    if time_npy is not None:
        print(f"  time.npy shape: {time_npy.shape}")

    # ---- 转换为 DataFrame ----
    print("\n[3/4] 转换数据格式...")

    # 检查是否有 domain_info
    domain_info = None
    norm_params_path = os.path.join(result_dir, 'kepler_vis', 'norm_params.json')
    if os.path.exists(norm_params_path):
        with open(norm_params_path, 'r') as f:
            norm_params = json.load(f)
        domain_info = {
            'channel': norm_params.get('domain_channel', -1),
            'month': norm_params.get('domain_month', -1),
            'name': norm_params.get('domain_name', '')
        }

    df = predictions_to_dataframe(
        predict_npy, label_npy, time_npy,
        station_coords, pm25_mean, pm25_std,
        hist_len=hist_len, pred_len=pred_len,
        domain_info=domain_info
    )

    if len(df) == 0:
        print("❌ DataFrame 为空，请检查站点坐标与预测结果的匹配关系")
        print("   可能原因:")
        print("   1. 站点坐标文件列名不正确 → 运行 diagnose_csv() 诊断")
        print("   2. 预测值维度与站点数量不匹配")
        print("   3. 经纬度数值格式错误（非数字）")
        return None

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(result_dir, 'kepler_vis')
    os.makedirs(output_dir, exist_ok=True)

    # 保存 CSV
    if save_csv:
        csv_path = os.path.join(output_dir, 'pm25_predictions.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 数据已保存到: {csv_path}")

    # 生成统计摘要
    generate_summary(df, output_dir)

    # ---- 创建 kepler.gl 地图 ----
    print("\n[4/4] 创建 Kepler.gl 地图...")
    try:
        map_1 = KeplerGl(height=700, data={'PM2.5 Predictions': df.copy()})
        print(f"✅ Kepler.gl 地图创建成功")
    except Exception as e:
        print(f"❌ Kepler.gl 地图创建失败: {e}")
        # 仍然保存 CSV
        return None

    # 应用配置
    try:
        kepler_config = get_kepler_config()
        map_1.config = kepler_config
    except Exception as e:
        print(f"⚠️ 配置应用失败（使用默认配置）: {e}")

    # 保存 HTML
    if save_html:
        html_path = os.path.join(output_dir, 'pm25_kepler_map.html')
        try:
            map_1.save_to_html(file_name=html_path)
            print(f"✅ 地图已保存到: {html_path}")
            _inject_china_map_tiles(html_path)
        except Exception as e:
            print(f"❌ HTML 保存失败: {e}")

    return map_1


def visualize_domain_comparison(all_results, station_coords_file,
                                 pm25_mean, pm25_std,
                                 output_dir='./kepler_domain_comparison',
                                 hist_len=1, pred_len=24):
    """
    多 Domain 对比可视化

    参数:
        all_results: 列表，每个元素为字典 {'domain': (i, j), 'result_dir': str}
        station_coords_file: 站点坐标文件
        pm25_mean, pm25_std: 归一化参数
        output_dir: 输出目录
        hist_len, pred_len: 时间窗口参数
    """
    print(f"\n{'='*60}")
    print(f"多 Domain 对比可视化")
    print(f"共 {len(all_results)} 个 Domain")
    print(f"{'='*60}")

    station_coords = load_station_coords(coords_file=station_coords_file)
    if len(station_coords) == 0:
        print("❌ 站点坐标加载失败")
        return None

    all_dfs = []
    success_count = 0
    for result_info in all_results:
        i, j = result_info['domain']
        result_dir = result_info['result_dir']

        predict_path = os.path.join(result_dir, 'predict.npy')
        label_path = os.path.join(result_dir, 'label.npy')

        if not os.path.exists(predict_path):
            continue

        predict_npy = np.load(predict_path)
        label_npy = np.load(label_path)
        time_npy = np.load(os.path.join(result_dir, 'time.npy')) if os.path.exists(os.path.join(result_dir, 'time.npy')) else None

        domain_info = {
            'channel': i,
            'month': j,
            'name': f'meteo{i}_month{j}'
        }

        df = predictions_to_dataframe(
            predict_npy, label_npy, time_npy,
            station_coords, pm25_mean, pm25_std,
            hist_len=hist_len, pred_len=pred_len,
            domain_info=domain_info
        )

        if len(df) > 0:
            all_dfs.append(df)
            success_count += 1

    if len(all_dfs) == 0:
        print("❌ 没有有效的 Domain 数据")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'pm25_all_domains.csv')
    combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 合并数据已保存: {csv_path}")

    try:
        map_1 = KeplerGl(height=700, data={'PM2.5 All Domains': combined_df.copy()})
        html_path = os.path.join(output_dir, 'pm25_domain_comparison.html')
        map_1.save_to_html(file_name=html_path)
        _inject_china_map_tiles(html_path)
        print(f"✅ Domain 对比地图已保存: {html_path}")
        print(f"   共 {len(combined_df)} 条记录，覆盖 {success_count}/{len(all_results)} 个 Domain")
        return map_1
    except Exception as e:
        print(f"❌ Domain 对比地图创建失败: {e}")
        return None
    
def _inject_china_map_tiles(html_path):
    """
    在保存的 Kepler.gl HTML 中注入高德地图瓦片，解决国内黑屏问题。
    国内网络加载 Mapbox 瓦片困难，替换为高德瓦片。
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    if 'CHINA_MAP_TILES_INJECTED' in html_content:
        return

    inject_script = """
<script>
// CHINA_MAP_TILES_INJECTED
(function(){
  var S={
    amap:{version:8,sources:{"amap-tiles":{type:"raster",tiles:["https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}","https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}","https://webrd03.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"],tileSize:256}},layers:[{id:"amap-layer",type:"raster",source:"amap-tiles",minzoom:0,maxzoom:18}]},
    osm:{version:8,sources:{"osm-tiles":{type:"raster",tiles:["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png","https://b.tile.openstreetmap.org/{z}/{x}/{y}.png","https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"],tileSize:256}},layers:[{id:"osm-layer",type:"raster",source:"osm-tiles",minzoom:0,maxzoom:18}]}
  };
  function gm(){var m=document.querySelector('.mapboxgl-map');if(m&&m._map)return m._map;for(var k in window){try{if(window[k]&&window[k].getStyle&&window[k].setStyle&&window[k].getCanvas)return window[k]}catch(e){}}return null}
  window.switchToAmap=function(){var m=gm();if(m)m.setStyle(S.amap)};
  window.switchToOSM=function(){var m=gm();if(m)m.setStyle(S.osm)};
  var cp=document.createElement('div');
  cp.style.cssText='position:fixed;bottom:20px;left:10px;z-index:9999;background:white;padding:10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:sans-serif;';
  cp.innerHTML='<b>底图切换</b><br><button onclick=switchToAmap() style="margin:2px;padding:6px 12px;cursor:pointer">高德地图</button><button onclick=switchToOSM() style="margin:2px;padding:6px 12px;cursor:pointer">OpenStreetMap</button><div style="font-size:11px;color:#999;margin-top:4px">黑屏请点击按钮切换底图</div>';
  document.body.appendChild(cp);
  setTimeout(function(){var m=gm();if(m)m.setStyle(S.amap)},8000);
})();
</script>
"""

    if '</body>' in html_content:
        html_content = html_content.replace('</body>', inject_script + '\n</body>')
    else:
        html_content += inject_script

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
