import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from model.industry_models import *
from utils.industry_utils import *
import matplotlib.pyplot as plt
# ---------------- Dataset & collate ----------------
class DownHierDataset(Dataset):
    def __init__(self, chunks_list, targets):
        self.chunks_list = chunks_list
        self.targets = targets
    def __len__(self):
        return len(self.chunks_list)
    def __getitem__(self, idx):
        return self.chunks_list[idx], self.targets[idx]

def collate_fn(batch):
    seqs_chunks, targets = zip(*batch)
    B = len(seqs_chunks)
    chunk_counts = [len(x) for x in seqs_chunks]
    max_chunks = max(chunk_counts) if len(chunk_counts) > 0 else 0
    max_chunk_len = max([max([c.shape[0] for c in seq]) if len(seq)>0 else 0 for seq in seqs_chunks])
    if max_chunk_len == 0:
        raise ValueError("zero-length chunk detected")
    F = seqs_chunks[0][0].shape[1]
    padded = torch.zeros((B, max_chunks, max_chunk_len, F), dtype=torch.float32)
    chunk_lengths = torch.zeros((B, max_chunks), dtype=torch.long)
    for i, seq in enumerate(seqs_chunks):
        for j, chunk in enumerate(seq):
            L = chunk.shape[0]
            padded[i, j, :L, :] = torch.from_numpy(chunk)
            chunk_lengths[i, j] = L
    targets = torch.tensor(np.vstack(targets), dtype=torch.float32)
    return padded, chunk_lengths, torch.tensor(chunk_counts, dtype=torch.long), targets