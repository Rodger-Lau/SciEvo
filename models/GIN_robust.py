import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool


class RobustGIN(nn.Module):
    """Small, bias-resistant GIN backbone for binary molecular classification."""

    def __init__(
        self,
        in_channels=7,
        hidden_dim=64,
        output_dim=1,
        num_layers=3,
        dropout=0.3,
        drop_out=None,
        normalize_features=True,
        use_size_feature=True,
    ):
        super().__init__()
        if drop_out is not None:
            dropout = drop_out

        self.num_layers = num_layers
        self.drop_out = dropout
        self.normalize_features = normalize_features
        self.use_size_feature = use_size_feature
        self.act = nn.ReLU()
        self.register_buffer("feature_scale", self._make_feature_scale(in_channels))

        self.initial_lin = nn.Linear(in_channels, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        readout_dim = hidden_dim * 2 + (1 if use_size_feature else 0)
        self.readout_norm = nn.LayerNorm(readout_dim)
        self.classifier1 = nn.Linear(readout_dim, hidden_dim)
        self.classifier2 = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def _make_feature_scale(in_channels):
        scale = torch.ones(in_channels, dtype=torch.float32)
        molecular_integer_scales = torch.tensor(
            [20.0, 4.0, 4.0, 1.0, 4.0],
            dtype=torch.float32,
        )
        usable = min(in_channels, molecular_integer_scales.numel())
        scale[:usable] = molecular_integer_scales[:usable]
        return scale

    def _unpack_inputs(self, data, edge_index=None, batch=None):
        if hasattr(data, "x"):
            return data.x, data.edge_index, getattr(data, "batch", None)
        return data, edge_index, batch

    def forward(self, data, edge_index=None, batch=None, activate_num=0):
        if hasattr(data, "x") and edge_index is not None and batch is None and not torch.is_tensor(edge_index):
            activate_num = edge_index
        x, edge_index, batch = self._unpack_inputs(data, edge_index, batch)
        x = x.float()
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if self.normalize_features:
            x = x / self.feature_scale.to(x.device).clamp_min(1.0)

        positive_count = 0
        activation_denominator = 0

        x = self.initial_lin(x)
        x = self.input_norm(x)
        x = self.act(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            h = conv(x, edge_index)
            h = norm(h)
            h = self.act(h)
            h = F.dropout(h, p=self.drop_out, training=self.training)
            x = residual + h

            positive_count += (x > 0).sum().item()
            activation_denominator += x.numel()

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        readout_parts = [mean_pool, max_pool]
        if self.use_size_feature:
            node_counts = torch.bincount(batch, minlength=mean_pool.size(0)).float().to(x.device)
            size_feature = torch.log1p(node_counts).unsqueeze(-1) / 5.0
            readout_parts.append(size_feature)
        graph_emb = torch.cat(readout_parts, dim=-1)

        graph_emb = self.readout_norm(graph_emb)
        graph_emb = self.classifier1(graph_emb)
        graph_emb = self.act(graph_emb)

        positive_count += (graph_emb > 0).sum().item()
        activation_denominator += graph_emb.numel()

        graph_emb = F.dropout(graph_emb, p=self.drop_out, training=self.training)
        logits = self.classifier2(graph_emb).view(-1)

        if activation_denominator > 0:
            activate_num += positive_count / activation_denominator

        return logits, activate_num


GIN = RobustGIN
