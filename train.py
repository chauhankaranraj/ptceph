import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model import LSTMPredictor
from data_utils import BackblazeSingleDriveDataset, bb_data_transform


# TODO: add preprocessing for backblaze data
if __name__ == "__main__":
    META_DIR = '/home/kachauha/Downloads/data_Q4_2018_serials/meta'
    DUMMY_DATA = False

    if DUMMY_DATA:
        # dummy dataset
        num_feats = 10
        num_classes = 2
        time_window = 2
        num_serials = 100

        # random vectors as input
        train_dataset = []
        for i in range(num_serials):
            curr_ts_len = random.randint(time_window, 4*time_window)
            train_dataset.append(torch.rand(size=(curr_ts_len, num_feats)))
        targets = torch.randint(num_classes, size=(num_serials,1))
    else:
        DATA_ROOT_DIR = '/home/kachauha/Downloads/data_Q4_2018_serials/failed'
        use_cols = list(pd.read_csv(os.path.join(META_DIR, 'means.csv'), header=None)[0]) + ['status']
        train_serials = ['ZA13Q5GK']

        # meta data
        num_classes = 3
        time_window = 6
        num_feats = len(use_cols) - 1
        num_serials = len(train_serials)

        # transforms
        df_to_tensor = lambda df: torch.Tensor(df.values)

        # create by chaining single serial datsets
        train_dataset = torch.utils.data.ChainDataset(
            BackblazeSingleDriveDataset(os.path.join(DATA_ROOT_DIR, serial + '.csv'),
                                        feat_cols=use_cols,
                                        time_window_size=time_window,
                                        transform=bb_data_transform,
                                        target_transform=lambda x: torch.LongTensor(x.values))
            for serial in train_serials
        )

    # training params
    batch_size = 1
    num_epochs = 10
    learning_rate = 0.01

    model = LSTMPredictor(num_feats, num_classes)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    for epoch in range(num_epochs):
        for seq, label in train_dataset:
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
