import torch
import torch.nn as nn


class LSTMtoy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTMtoy, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, 2)
        self.lstm2 = nn.LSTM(128, 256, 3)
        # self.lstm1 = nn.LSTMCell(1, 51)
        # self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(256, output_dim)

    def forward(self, x):
        output, hidden = self.lstm1(x)
        output, hidden = self.lstm2(output)
        # take output of last lstm cell for the last time cel
        # and feed to linear layer
        logits = self.linear(output[-1, :])
        return logits

