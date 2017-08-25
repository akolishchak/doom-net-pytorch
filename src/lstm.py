#
# lstm.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.i2h_bn = nn.BatchNorm1d(4 * hidden_size)
        self.h2h_bn = nn.BatchNorm1d(4 * hidden_size)
        self.cx_bn = nn.BatchNorm1d(hidden_size)

    def forward(self, input, cell):
        hx, cx = cell
        input = self.i2h_bn(self.i2h(input)) + self.h2h_bn(self.h2h(hx))
        gates = F.sigmoid(input[:, :3*self.hidden_size])
        in_gate = gates[:, :self.hidden_size]
        forget_gate = gates[:, self.hidden_size:2*self.hidden_size]
        out_gate = gates[:, 2*self.hidden_size:3*self.hidden_size]
        input = F.tanh(input[:, 3*self.hidden_size:4*self.hidden_size])
        cx = (forget_gate * cx) + (in_gate * input)
        hx = out_gate * F.tanh(self.cx_bn(cx))
        return hx, cx
