import os
import json
import random
import datetime
from os.path import join as ospj
from itertools import chain, cycle

import numpy as np
import dask.dataframe as dd
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
from datasets import BackblazeSingleDrivePtDataset, get_train_test_serials


# TODO: add weighted loss
# TODO: if tranposing on cpu is expensive, then do that in gpu (not in transform funcs)
if __name__ == "__main__":
    # get args
    args = parse_train_args()

    # cpu or gpu
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    # split data into dirs
    WORK_DIR = ospj(args.data_root_dir, 'working')
    FAIL_DIR = ospj(args.data_root_dir, 'failed')
    META_DIR = ospj(args.data_root_dir, 'meta')

    # get metadata for scaling - moved this outside fn so that it's not read again and again
    means = torch.load(ospj(META_DIR, 'means.pt'))
    stds = torch.load(ospj(META_DIR, 'stds.pt'))
    # FIXME: this should not be needed. just drop cols where std==0
    stds[stds==0] = 1

    # transforms on raw data
    # NOTE: labels needs to be (batch) not (batch, 1). keep only last day data
    data_transform = lambda x: torch.as_tensor((x - means) / stds, dtype=torch.float32)
    label_transform = lambda x: torch.as_tensor(x[-1,...], dtype=torch.int64).squeeze(-1)

    # meta data
    num_classes = 3
    time_window = 6
    num_feats = means.shape[-1]
    class_labels = [i for i in range(num_classes)]

    # get serial numbers file paths to be used for train and test
    train_ser_files, test_ser_files = get_train_test_serials(WORK_DIR, FAIL_DIR, test_size=0.1)

    # create by chaining single serial datsets
    train_dataset = torch.utils.data.ChainDataset(
        BackblazeSingleDrivePtDataset(serfile,
                                    time_window_size=time_window,
                                    transform=data_transform,
                                    target_transform=label_transform)
        for serfile in train_ser_files
    )
    test_dataset = torch.utils.data.ChainDataset(
        BackblazeSingleDrivePtDataset(serfile,
                                    time_window_size=time_window,
                                    transform=data_transform,
                                    target_transform=label_transform)
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

    # get total number of test sequences
    # this will be used for init-ing tensors to store preds and labels during val/test
    num_test_pts = 0
    for ser_ds in test_dataset.datasets:
        num_test_pts += len(ser_ds)

    # init tensor to store labels. init now so that mem is not malloced/freed every iter
    all_test_preds = torch.empty(size=(num_test_pts, 1), device=device, dtype=torch.int64)
    all_test_labels = torch.empty(size=(num_test_pts, 1), device='cpu', dtype=torch.int64)
    for test_batch_idx, (_, test_labels) in enumerate(test_loader):
        all_test_labels[test_batch_idx*args.batch_size: (test_batch_idx+1)*args.batch_size] = test_labels

    # init model, loss, optimizer
    model = getattr(models, args.model_arch)(num_feats, num_classes)
    model = model.to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    if args.log_summaries:
        # save training params
        with open(ospj(args.log_dir, 'args_{}.json'.format(args.timestamp)), 'w') as f:
            json.dump(vars(args), f)

        # init summary writers
        train_writer = SummaryWriter(log_dir=ospj(args.log_dir,
                                                    '{}_{}'.format(model._get_name(), args.timestamp),
                                                    "train_0"))
        val_writer = SummaryWriter(log_dir=ospj(args.log_dir,
                                                    '{}_{}'.format(model._get_name(), args.timestamp),
                                                    "val_0"))
        global_step = 0

    for epoch in range(args.num_epochs):
        for batch_idx, (seqs, labels) in enumerate(train_loader):
            # udpate progress bar
            print("Epoch {:2d} Batch {:6d}".format(epoch, batch_idx), end='\r')

            # move to train device
            seqs, labels = seqs.to(device), labels.to(device)

            # reset for batch
            model.zero_grad()
            optimizer.zero_grad()

            # feed forward
            # need to make (seq_len, batch, input_size) from (batch, seq_len, input_size)
            log_probs = F.log_softmax(model(seqs.transpose(0, 1)), dim=1)

            # backprop. NOTE: labels needs to be (batch) not (batch, 1)
            # TODO: move this to transform fn
            loss = loss_function(log_probs, labels)

            # print output every so intervals
            if batch_idx % args.log_interval == 0:
                # get train and validation set metrics
                model.eval()
                with torch.no_grad():
                    for test_batch_idx, (test_seqs, _) in enumerate(test_loader):
                        print("Batch {:6d} / {:6d}".format(test_batch_idx, num_test_pts//args.batch_size), end='\r')

                        # need to move to compute device
                        test_seqs = test_seqs.to(device)

                        # save preds for current batch. NOTE: assumes all test labels are on cpu
                        all_test_preds[test_batch_idx*args.batch_size: (test_batch_idx+1)*args.batch_size] = \
                            torch.argmax(model(test_seqs.transpose(0, 1)), dim=1, keepdim=True)
                    prec, rec, f1, _ = precision_recall_fscore_support(all_test_preds.cpu(), all_test_labels, labels=class_labels)
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

                print('Epoch: {:2d} Batch: {:6d}\tLoss: {:.6f} Prec={:.2f}, {:.2f}, {:.2f} Rec={:.2f}, {:.2f}, {:.2f} F1={:.2f}, {:.2f}, {:.2f}'.format(
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

    # save the model only if explicitly told to
    if args.save_model:
        torch.save(model.state_dict(),
                    ospj(args.save_dir, '{}_{}.pt'.format(model._get_name(), args.timestamp)))
