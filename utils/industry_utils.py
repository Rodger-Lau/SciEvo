import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from model.industry_models import *
import matplotlib.pyplot as plt
# ---------------- util functions ----------------
# 使用训练时的scaler
def scale_chunks_list(chunks_list, scaler):
    out = []
    for seq_chunks in chunks_list:
        out.append([scaler.transform(ch) for ch in seq_chunks])
    return out
def add_gaussian_noise(sequences, noise_std=0.01, seed=42, columns=None):
    """
    为序列数据添加均值为0、方差可调的高斯噪声
    
    Args:
        sequences: list of arrays, shape [T_i, F] for each sequence
        noise_std: 噪声的标准差
        seed: 随机种子
        columns: None或列索引列表，None表示对所有列加噪声，否则只对指定列加噪声
        
    Returns:
        noisy_sequences: 添加噪声后的序列列表
    """
    if noise_std <= 0:
        return sequences
    
    rng = np.random.RandomState(seed)
    noisy_sequences = []
    
    for seq in sequences:
        if columns is None:
            # 对所有列加噪声
            noise = rng.normal(0, noise_std, size=seq.shape).astype(np.float32)
            noisy_seq = seq + noise
        else:
            # 只对指定列加噪声
            noisy_seq = seq.copy()
            T, F = seq.shape
            for col_idx in columns:
                if 0 <= col_idx < F:
                    noise = rng.normal(0, noise_std, size=(T,)).astype(np.float32)
                    noisy_seq[:, col_idx] += noise
            noisy_seq = noisy_seq.astype(np.float32)
        
        noisy_sequences.append(noisy_seq)
    
    return noisy_sequences

def downsample_avg(seq, window):
    if window <= 1:
        return seq.copy()
    L, F = seq.shape
    newL = L // window
    if newL == 0:
        return seq.copy()
    trimmed = seq[: newL * window]
    return trimmed.reshape(newL, window, F).mean(axis=1)

def split_into_chunks(seq, chunk_size):
    T, F = seq.shape
    chunks = []
    num = (T + chunk_size - 1) // chunk_size
    for i in range(num):
        start = i * chunk_size
        end = min((i+1) * chunk_size, T)
        chunks.append(seq[start:end])
    return chunks

def expand_chunk_weights_to_steps(seq_len, chunk_size, chunk_weights):
    w = np.zeros(seq_len, dtype=np.float32)
    num_chunks = int(np.ceil(seq_len / float(chunk_size)))
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, seq_len)
        if i < len(chunk_weights):
            w[start:end] = chunk_weights[i]
    # normalize to [0,1]
    if np.max(w) > np.min(w):
        w = (w - np.min(w)) / (np.max(w) - np.min(w) + 1e-8)
    return w

def plot_colored_series(values, weights, out_path, title="Sequence with importance", cmap="viridis"):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    T = len(values)
    x = np.arange(T, dtype=np.float32)
    points = np.array([x, values], dtype=np.float32).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_weights = (weights[:-1] + weights[1:]) / 2.0 if T > 1 else np.array([weights[0]], dtype=np.float32)
    norm = Normalize(vmin=0.0, vmax=1.0)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_weights)
    lc.set_linewidth(2.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    if T > 0:
        rng = values.max() - values.min()
        ax.set_ylim(values.min() - 0.05 * (rng + 1e-8), values.max() + 0.05 * (rng + 1e-8))
    ax.set_title(title)
    ax.set_xlabel("Time index (downsampled)")
    ax.set_ylabel("Value")
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Importance")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)