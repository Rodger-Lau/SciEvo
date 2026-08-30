import torch
from torch import nn
from model_air.cells import LSTMCell


class LSTM(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, out_dim, task_index=0):
        super(LSTM, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = out_dim
        self.task_index = task_index
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.lstm_cell = LSTMCell(self.hid_dim, self.hid_dim)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        cn = c0
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            # x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            if self.task_index == 1:
                x = torch.cat((xn.argmax(dim=-1, keepdim=True).float(), feature[:, self.hist_len+i]), dim=-1)
            elif self.task_index == 0:
                x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            x = self.fc_in(x)
            hn, cn = self.lstm_cell(x, (hn, cn))
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            if self.task_index == 1:
                # xn = torch.tensor(xn.argmax(dim=-1).unsqueeze(dim=-1), dtype=torch.float32, requires_grad=True)
                pass
            pm25_pred.append(xn)
        pm25_pred = torch.stack(pm25_pred, dim=1)
        return pm25_pred
