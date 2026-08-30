import torch
from torch import nn
from model_air.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid


class nodesFC_GRU(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, out_dim=1, task_index=0):
        super(nodesFC_GRU, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = out_dim
        self.graph_mlp_out = 1
        self.task_index = task_index
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.gru_cell = GRUCell(self.in_dim + self.graph_mlp_out, self.hid_dim)
        self.graph_mlp = Sequential(Linear(self.city_num * self.in_dim, self.city_num * self.graph_mlp_out),
                                   Sigmoid())

    def forward(self, pm25_hist, feature):
        total_activate_freq = 0
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            if self.task_index == 1:
                x = torch.cat((xn.argmax(dim=-1, keepdim=True).float(), feature[:, self.hist_len+i]), dim=-1)
            elif self.task_index == 0:
                x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            # nodes FC
            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = xn_gnn.view(self.batch_size, -1)
            xn_gnn = self.graph_mlp(xn_gnn)
            xn_gnn = xn_gnn.view(self.batch_size, self.city_num, 1)
            x = torch.cat([xn_gnn, x], dim=-1)
            # nodes FC
            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            count = (xn > 0).sum().item()
            size = 1
            for j in xn.shape:
                size *= j
            activate_freq = count / size
            total_activate_freq += activate_freq
            xn = self.fc_out(xn)
            # print(f'xn:{xn}')
            # if self.task_index == 1:
            #     xn = torch.tensor(xn.argmax(dim=-1).unsqueeze(dim=-1), dtype=torch.float32, requires_grad=True)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)
        activate_freq = total_activate_freq / self.pred_len
        return pm25_pred, activate_freq
