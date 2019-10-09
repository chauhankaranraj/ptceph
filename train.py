import os
from os.path import join as ospj
import random
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import models
from models import LSTMtoy
from arg_parsers import parse_train_args
from data_utils import BackblazeSingleDriveDataset, bb_data_transform


if __name__ == "__main__":
    # get args
    args = parse_train_args()

    # cpu or gpu
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    # split data into dirs
    WORK_DIR = ospj(args.data_root_dir, 'working')
    FAIL_DIR = ospj(args.data_root_dir, 'failed')
    META_DIR = ospj(args.data_root_dir, 'meta')

    # columns used as features and target
    feat_cols = list(pd.read_csv(ospj(META_DIR, 'means.csv'), header=None)[0])
    target_col = ['status']
    train_ser_files = [f for f in os.listdir(FAIL_DIR) if os.path.isfile(ospj(FAIL_DIR, f))]

    # meta data
    num_classes = 3
    time_window = 6
    num_feats = len(feat_cols)
    num_serials = len(train_ser_files)

    # transforms. TODO: make this a proper function
    df_to_tensor = lambda df: torch.Tensor(df.values)

    # create by chaining single serial datsets
    train_dataset = torch.utils.data.ChainDataset(
        BackblazeSingleDriveDataset(ospj(FAIL_DIR, serfile),
                                    feat_cols=feat_cols,
                                    target_cols=target_col,
                                    time_window_size=time_window,
                                    transform=bb_data_transform,
                                    target_transform=lambda x: torch.LongTensor(x.values))
        for serfile in train_ser_files
    )

    # init model, loss, optimizer
    model = getattr(models, args.model_arch)(num_feats, num_classes)
    model = model.to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    for epoch in range(args.num_epochs):
        for seq, label in train_dataset:
            # move to train device
            seq, label = seq.to(device), label.to(device)

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

    # save the model
    torch.save(model.state_dict(),
                ospj(args.save_dir, '{}_{}.pt'.format(model._get_name(), args.timestamp)))
