import os
import datetime
import argparse
import multiprocessing
from os.path import join as ospj

import torch


def parse_train_args():
    parser = argparse.ArgumentParser(description="Arguments used for training drive failure prediction model")

    parser.add_argument(
        "-ne", "--num-epochs",
        type=int,
        default=5,
        required=False,
        help="Number of epochs to train the model for"
    )

    parser.add_argument(
        "-bs", "--batch-size",
        type=int,
        default=1,
        required=False,
        help="Number of samples (of frames) per batch"
    )

    parser.add_argument(
        "-lr", "--learning-rate",
        type=float,
        default=0.001,
        required=False,
        help="Learning rate of model"
    )

    # FIXME: will see use_cuda=True when setting -uc=False from cmdline
    parser.add_argument(
        "-uc", "--use-cuda",
        type=bool,
        default=torch.cuda.is_available(),
        required=False,
        help="Uses GPU for training if set to 1 else uses cpu (set to 0)"
    )

    parser.add_argument(
        "-cc", "--cpu-cores",
        type=int,
        default=multiprocessing.cpu_count(),
        required=False,
        help="How many CPU cores are to be used as workers for data loading / preprocessing\
            wherever multicore processing is possible"
    )

    parser.add_argument(
        "-ma", "--model-arch",
        type=str,
        default="LSTMtoy",
        required=False,
        help="Name of the model architecture to train"
    )

    parser.add_argument(
        "-sm", "--save-model",
        action='store_true',
        help="Saves parameters of model after training if set to true"
    )

    parser.add_argument(
        "-lm", "--load-model",
        action='store_true',
        help="Loads parameters of model from a saved file from a previous training session if set to true"
    )

    parser.add_argument(
        "-lf", "--load-fname",
        type=str,
        default='',
        required=False,
        help="Name of the file from which model parameters are to be loaded"
    )

    parser.add_argument(
        "-li", "--log-interval",
        type=int,
        default=1,
        required=False,
        help="Number of batches after which to writes tensorboard summaries"
    )

    parser.add_argument(
        "-ls", "--log-summaries",
        action='store_true',
        help="Writes tensorboard summaries to a directory if set to true"
    )

    parser.add_argument(
        "-ts", "--timestamp",
        type=str,
        default=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
        required=False,
        help="Timestamp associated with current training session. This will be used to identify logs and saved model params"
    )

    # if the default dir does not exist, create it
    DEFAULT_SAVE_DIR = ospj(os.getcwd(), 'pretrained')
    if not os.path.exists(DEFAULT_SAVE_DIR):
        os.makedirs(DEFAULT_SAVE_DIR)

    parser.add_argument(
        "-sd", "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        required=False,
        help="Directory in which model parameters will be saved"
    )

    # if the default dir does not exist, create it
    DEFAULT_LOG_DIR = ospj(os.getcwd(), 'logs')
    if not os.path.exists(DEFAULT_LOG_DIR):
        os.makedirs(DEFAULT_LOG_DIR)

    parser.add_argument(
        "-ld", "--log-dir",
        type=str,
        default=DEFAULT_LOG_DIR,
        required=False,
        help="Directory in which tensorboard train summaries will be written to"
    )

    # if the default dir does not exist, create it
    DEFAULT_DATA_ROOT_DIR = '/home/kachauha/Downloads/data_Q4_2018_serials'
    if not os.path.exists(DEFAULT_DATA_ROOT_DIR):
        raise FileNotFoundError('data root dir not passed as cmd line input.\
             the default path also does not exist')

    parser.add_argument(
        "-rd", "--data-root-dir",
        type=str,
        default=DEFAULT_DATA_ROOT_DIR,
        required=False,
        help="Directory containing the dataset"
    )

    return parser.parse_args()
