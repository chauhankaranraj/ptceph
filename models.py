import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMtoy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTMtoy, self).__init__()
        self.lstm = nn.LSTM(input_dim, 256, 5)
        # self.lstm1 = nn.LSTMCell(1, 51)
        # self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(256, output_dim)

    def forward(self, x):
        output, hidden = self.lstm(x)
        # take output of last lstm cell for the last time cel
        # and feed to linear layer
        log_probs = F.log_softmax(self.linear(output[-1, :]))
        return log_probs