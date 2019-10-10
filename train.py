import os
from os.path import join as ospj
import random
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import models
from models import LSTMtoy
from arg_parsers import parse_train_args
from data_utils import BackblazeSingleDriveDataset


# TODO: if tranposing on cpu is not intensive, then do that in transform
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

    # split into train and test files
    ser_files = [f for f in os.listdir(FAIL_DIR) if os.path.isfile(ospj(FAIL_DIR, f))]
    train_ser_files, test_ser_files = train_test_split(ser_files, test_size=0.1)

    # meta data
    num_classes = 3
    time_window = 6
    num_feats = len(feat_cols)
    num_serials = len(train_ser_files)
    class_labels = [i for i in range(num_classes)]

    def bb_data_transform(df):
        # scale from bytes to gigabyte
        # FIXME: this is done coz mean centering does not work w/ large numbers
        df['capacity_bytes'] /= 10**9

        # get metadata for scaling
        means = pd.read_csv(ospj(META_DIR, 'means.csv'), header=None).set_index(0).transpose()
        stds = pd.read_csv(ospj(META_DIR, 'stds.csv'), header=None).set_index(0).transpose()

        # 0 mean, 1 std
        # FIXME: subtract and divide w/out using index else nans
        df = (df - means.values)

        # FIXME: divide by zero error
        if (stds.values==0).any():
            # print('DivideByZeroWarning: std has 0. Dividing will result in nans. Replacing with 1\'s')
            stds = stds.replace(to_replace=0, value=1)
        df = df / stds.values

        # to tensor
        return torch.Tensor(df.values)

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
    test_dataset = torch.utils.data.ChainDataset(
        BackblazeSingleDriveDataset(ospj(FAIL_DIR, serfile),
                                    feat_cols=feat_cols,
                                    target_cols=target_col,
                                    time_window_size=time_window,
                                    transform=bb_data_transform,
                                    target_transform=lambda x: torch.LongTensor(x.values))
        for serfile in test_ser_files
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                shuffle=False,
                                                batch_size=args.batch_size,
                                                num_workers=args.cpu_cores,
                                                )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                shuffle=False,
                                                batch_size=args.batch_size,
                                                num_workers=args.cpu_cores,
                                                )

    # init model, loss, optimizer
    model = getattr(models, args.model_arch)(num_feats, num_classes)
    model = model.to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    if args.log_summaries:
        train_writer = SummaryWriter(log_dir=ospj(args.log_dir,
                                                    '{}_{}'.format(model._get_name(), args.timestamp),
                                                    "train_0"))
        val_writer = SummaryWriter(log_dir=ospj(args.log_dir,
                                                    '{}_{}'.format(model._get_name(), args.timestamp),
                                                    "val_0"))
        global_step = 0

    for epoch in range(args.num_epochs):
        for batch_idx, (seq, label) in enumerate(train_loader):
            # move to train device
            seq, label = seq.to(device), label.to(device)

            # reset for batch
            model.zero_grad()
            optimizer.zero_grad()

            # feed forward
            # need to make (seq_len, batch, input_size) from (batch, seq_len, input_size)
            log_probs = F.log_softmax(model(seq.transpose(0, 1)), dim=1)

            # backprop. NOTE: label needs to be (batch) not (batch, 1)
            # TODO: move this to transform fn
            loss = loss_function(log_probs, label.squeeze(-1))

            # print output every so intervals
            if batch_idx % args.log_interval == 0:
                # get train and validation set metrics
                model.eval()
                with torch.no_grad():
                    all_preds = None
                    all_labels = None
                    for test_seq, test_label in test_loader:
                        # need to move to compute device
                        test_seq = test_seq.to(device)

                        # get predictions TODO: remove if-else
                        if all_preds is None:
                            all_preds = torch.argmax(model(test_seq.transpose(0, 1)), dim=1)
                            all_labels = test_label
                        else:
                            all_preds = torch.cat((all_preds, torch.argmax(model(test_seq.transpose(0, 1)), dim=1)))
                            all_labels = torch.cat((all_labels, test_label))
                    prec, rec, f1, _ = precision_recall_fscore_support(all_preds.cpu(), all_labels, labels=class_labels)
                model.train()

                # TODO: how to get prec, rec, f1 for train set
                if args.log_summaries:
                    train_writer.add_scalar('loss', loss.item(), global_step)
                    # NOTE: assumes prec, rec, f1 are of same length
                    for i in range(len(prec)):
                        val_writer.add_scalar('confmat/precision_{}'.format(i), prec[i], global_step)
                        val_writer.add_scalar('confmat/recall_{}'.format(i), rec[i], global_step)
                        val_writer.add_scalar('confmat/f1_score_{}'.format(i), f1[i], global_step)
                    global_step += 1

                print('Epoch: {:2d} Batch: {:2d}\tLoss: {:.6f} Prec={:.2f}, {:.2f}, {:.2f} Rec={:.2f}, {:.2f}, {:.2f} F1={:.2f}, {:.2f}, {:.2f}'.format(
                        epoch,
                        batch_idx,
                        loss.item(),
                        *prec,
                        *rec,
                        *f1
                        )
                )
            loss.backward()
            optimizer.step()

    # save the model
    torch.save(model.state_dict(),
                ospj(args.save_dir, '{}_{}.pt'.format(model._get_name(), args.timestamp)))
