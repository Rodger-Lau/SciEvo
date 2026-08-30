import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid


class MLP(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, out_dim=1, task_index=0):
        super(MLP, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.hid_dim = 16
        self.out_dim = out_dim
        self.graph_mlp_out = 1
        self.graph_mlp_hid = 1
        self.task_index = task_index
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.mlp = Sequential(Linear(self.hid_dim, self.hid_dim),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    Linear(self.hid_dim, self.hid_dim),
                                    nn.LeakyReLU(negative_slope=0.1)
                                    )
    def forward(self, pm25_hist, feature):
        total_activate_freq = 0
        pm25_pred = []
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            if self.task_index == 1:
                x = torch.cat((xn.argmax(dim=-1, keepdim=True).float(), feature[:, self.hist_len+i]), dim=-1)
            elif self.task_index == 0:
                x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            # print(f'{i}_after_cat x shape: {x.shape}')
            # print('fc_in:', self.fc_in)
            x = self.fc_in(x)
            x = self.mlp(x)
            count = (x > 0).sum().item()
            size = 1
            for j in x.shape:
                size *= j
            activate_freq = count / size
            total_activate_freq += activate_freq
            xn = self.fc_out(x)
            if self.task_index == 1:
                # xn= torch.tensor(xn.argmax(dim=-1).unsqueeze(dim=-1), dtype=torch.float32, requires_grad=True)
                # xn = xn.argmax(dim=-1, keepdim=True).float()
                pass
            pm25_pred.append(xn)
        pm25_pred = torch.stack(pm25_pred, dim=1)
        activate_freq = total_activate_freq / self.pred_len
        return pm25_pred, activate_freq
