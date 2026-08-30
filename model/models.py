import torch.nn as nn
import torch
import copy

from model.layers import *

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class CMuST(nn.Module):
    """
    applies attention mechanisms both temporally and spatially within a dataset. It is particularly designed for scenarios
    where data has inherent spatial and temporal attributes,
    multi-dimensional time series forecasting.
    """
    def __init__(self, num_nodes, input_len=12, output_len=12, tod_size=48, obser_dim=3, output_dim=1, 
        d_obser=24, d_tod=24, d_dow=24, d_ts=12, d_s=12, d_t=48, d_p=72,
        self_atten_dim=24, cross_atten_dim=24, ffn_dim=256, n_heads=4, dropout=0.1, device='cuda'
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.tod_size = tod_size
        
        self.obser_dim = obser_dim
        self.output_dim = output_dim
        self.d_obser = d_obser
        self.d_tod = d_tod
        self.d_dow = d_dow
        self.d_s = d_s
        self.d_t = d_t
        self.d_p = d_p
        self.self_atten_dim = self_atten_dim
        self.cross_atten_dim = cross_atten_dim
        
        self.h_dim = d_obser + d_s + d_t + d_p
        self.n_heads = n_heads
        self.device = device
        self.obser_start = 0
        self.s_start = d_obser
        self.t_start = d_obser + d_s
        self.dropout = dropout
        # self.dropout_ori = dropout
        self.obser_mlp = nn.Linear(obser_dim, d_obser)
        self.timestamp_mlp = nn.Linear(6, d_ts)
        self.spatial_mlp = nn.Linear(2, d_s)
        self.temporal_mlp = nn.Linear(d_ts+d_dow+d_tod, d_t)
        self.tod_embedding = nn.Embedding(tod_size, d_tod)
        self.dow_embedding = nn.Embedding(7, d_dow)
        self.prompt = nn.Parameter(torch.empty(input_len, num_nodes, d_p))
        self.ffn_dim = ffn_dim
        self.output_mlp = nn.Linear(input_len * self.h_dim, output_len * output_dim)
        # self.output_mlp = nn.Linear(input_len * self.d_obser, output_len * output_dim)
        self.layers = clones(MSTI(self.obser_start, self.s_start, self.t_start, d_obser, d_s, d_t, self.h_dim,
                                  self_atten_dim, cross_atten_dim, ffn_dim, n_heads, self.dropout), N=1)
        # self.relu = nn.ReLU()
    def forward(self, x):
        # if self.dropout != self.dropout_ori:
        #     self.layers = clones(MSTI(self.obser_start, self.s_start, self.t_start, self.d_obser, self.d_s, self.d_t, self.h_dim,
        #                           self.self_atten_dim, self.cross_atten_dim, self.ffn_dim, self.n_heads, self.dropout).to(self.device), N=1)
        #     self.dropout_ori = self.dropout
        # extract features
        batch_size = x.shape[0]
        tod = x[..., 1]
        dow = x[..., 2]
        coor = x[..., 3:5]
        timestamp = x[..., 5:11]
        
        obser = x[..., : self.obser_dim]
        obser_emb = self.obser_mlp(obser)  # (batch_size, input_len, num_nodes, d_obser)
        h = [obser_emb]

        spatial_emb = self.spatial_mlp(coor) # (batch_size, input_len, num_nodes, d_s)
        h.append(spatial_emb)
        
        tod_emb = self.tod_embedding((tod * self.tod_size).long())  # (batch_size, input_len, num_nodes, d_tod)
        dow_emb = self.dow_embedding(dow.long())  # (batch_size, input_len, num_nodes, d_dow)
        timestamp_emb = self.timestamp_mlp(timestamp)   # (batch_size, input_len, num_nodes, d_ts)
        temporal_emb = self.temporal_mlp(torch.cat((timestamp_emb, tod_emb, dow_emb), dim=-1))   # (batch_size, input_len, num_nodes, d_t)
        h.append(temporal_emb)
        
        expanded_prompt = self.prompt.expand(size=(batch_size, *self.prompt.shape))
        h.append(expanded_prompt)
        
        x = torch.cat(h, dim=-1)  # (batch_size, input_len, num_nodes, h_dim)
        
        # MSTI
        for layer in self.layers:
            layer.dropout = self.dropout
            x = layer(x)
        
        # output
        out = x.transpose(1, 2).reshape(batch_size, self.num_nodes, self.input_len * self.h_dim)
        out = self.output_mlp(out).view(batch_size, self.num_nodes, self.output_len, self.output_dim)
        # out = self.relu(out)
        count = (out > 0).sum().item()
        size = 1
        for i in out.shape:
            size *= i
        activate_freq = count / size
        out = out.transpose(1, 2)  # (batch_size, output_len, num_nodes, output_dim)
        
        return out, activate_freq
