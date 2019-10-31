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


class LSTMExp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTMExp, self).__init__()
        self.lstm_layers = [
            nn.LSTM(input_dim, 256, 2),
            nn.LSTM(256, 512, 2),
            nn.LSTM(512, 1024, 1),
            nn.LSTM(1024, 512, 2),
            nn.LSTM(512, 256, 2)
        ]
        self.linear = nn.Linear(self.lstm_layers[-1].hidden_size, output_dim)

    def forward(self, x):
        # pass input to first lstm layers obj
        output, _ = self.lstm_layers[0](x)

        # pass output of previous lstm layers to all of the remaining ones
        for lstm in self.lstm_layers[1:]:
            output, _ = lstm(output)

        # feed output of last lstm cell for the last time cell to linear layer
        logits = self.linear(output[-1, :])
        return logits

