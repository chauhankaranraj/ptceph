import torch
import torch.nn as nn
import torch.optim as optim

import random

from model import LSTMPredictor

# TODO: replace dummy data with backblaze data
if __name__ == "__main__":
    # dummy dataset
    num_feats = 10
    num_classes = 2
    time_window = 2
    num_serials = 100

    # random vectors as input
    dataset = []
    for i in range(num_serials):
        curr_ts_len = random.randint(time_window, 4*time_window)
        dataset.append(torch.rand(size=(curr_ts_len, num_feats)))
    targets = torch.randint(num_classes, size=(num_serials,1))

    # training params
    batch_size = 1
    num_epochs = 10
    learning_rate = 0.01

    model = LSTMPredictor(num_feats, num_classes)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    for epoch in range(num_epochs):
        for seq,label in zip(dataset, targets):
            # reset for batch
            model.zero_grad()
            optimizer.zero_grad()

            # feed forward
            log_probs = model(seq.unsqueeze(1))

            # backprop
            loss = loss_function(log_probs, label)
            print("Loss = {:3.5f}".format(loss.item()))
            loss.backward()
            optimizer.step()
