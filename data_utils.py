import os
from glob import glob as gg
from os.path import join as ospj
from itertools import chain, cycle

import torch
import pandas as pd
from sklearn.model_selection import  train_test_split

from joblib import Parallel, delayed, parallel_backend


# # get metadata for scaling - moved this outside fn so that it's not read again and again
# means = pd.read_csv(ospj(META_DIR, 'means.csv'), header=None).set_index(0).transpose()
# stds = pd.read_csv(ospj(META_DIR, 'stds.csv'), header=None).set_index(0).transpose()
# def bb_data_transform(df):
#     global means, stds
#     # scale from bytes to gigabyte
#     # FIXME: this is done coz mean centering does not work w/ large numbers
#     df['capacity_bytes'] /= 10**9

#     # 0 mean, 1 std
#     # FIXME: subtract and divide w/out using index else nans
#     df = (df - means.values)

#     # FIXME: divide by zero error
#     if (stds.values==0).any():
#         # print('DivideByZeroWarning: std has 0. Dividing will result in nans. Replacing with 1\'s')
#         stds = stds.replace(to_replace=0, value=1)
#     df = df / stds.values

#     # to tensor
#     # NOTE: need to make (seq_len, batch, input_size) from (batch, seq_len, input_size)
#     return torch.Tensor(df.values)


def get_train_test_serials(work_dir, fail_dir, test_size=None, train_size=None):
    # sanitize inputs
    if test_size is None:
        if train_size is None:
            raise RuntimeError("Specify at least one of train_size or test_size")
        else:
            test_size = 1 - train_size

    # get serial numbers in each category
    failed_ser_files = [ospj(fail_dir, f) for f in os.listdir(fail_dir) if os.path.isfile(ospj(fail_dir, f))]
    working_ser_files = [ospj(work_dir, f) for f in os.listdir(work_dir) if os.path.isfile(ospj(work_dir, f))]

    # split each set (working, failed) into train and test files
    working_files_train, working_files_test = train_test_split(working_ser_files, test_size=test_size)
    failed_files_train, failed_files_test = train_test_split(failed_ser_files, test_size=test_size)

    # oversampling in train set to relax class imbalance somewhat
    if len(working_files_train) > len(failed_files_train):
        train_ser_files = list(chain(*zip(cycle(failed_files_train), working_files_train)))
    else:
        train_ser_files = list(chain(*zip(cycle(working_files_train), failed_files_train)))

    # dont oversample in test set - this will skew evaluation results
    test_ser_files = working_files_test + failed_files_test

    return train_ser_files, test_ser_files


# TODO: decide feat cols and label col
class BackblazeSingleDriveDataset(torch.utils.data.IterableDataset):
    """
    Dataset to iterate over data from a serial number, in chunks of time_window days

    This is an IterableDataset because that makes it easier for chaining.
    If it were a regular Dataset, then random reads would be possible which means
    each csv file could be read multiple times which can be expensive.
    """
    def __init__(self, fpath, feat_cols=None, target_cols=['status'], time_window_size=6, transform=None, target_transform=None):
        super(BackblazeSingleDriveDataset, self).__init__()

        # ensure data file exists
        assert os.path.exists(fpath)

        # time frame of data that will be consider one "sequence"
        self.time_window_size = time_window_size

        # metadata for processing
        self.feat_cols = feat_cols
        self.target_cols = target_cols

        # since it is a single serial file, it should fit in memory
        # FIXME: this is a hack to sort (assumes date column will be available in data)
        self._df = pd.read_csv(fpath, usecols=['date']+feat_cols+target_cols)
        self._df = self._df.sort_values(by='date', ascending=True)
        self._df = self._df.drop('date', axis=1)

        # current index in iteration
        self._curr_idx = 0

        # preprocessor
        self.transform = transform
        self.target_transform = target_transform

    def __iter__(self):
        while self._curr_idx < (len(self._df) - self.time_window_size + 1):
            # get current time chunk, transform if available
            idx_chunk = self._df.iloc[self._curr_idx: self._curr_idx + self.time_window_size, :]

            # split data and target
            data = idx_chunk.drop(self.target_cols, axis=1)
            target = idx_chunk[self.target_cols].iloc[-1]

            # transform what is necessary
            if self.transform is not None:
                data = self.transform(data)
            if self.target_transform is not None:
                target = self.target_transform(target)

            # return a tuple (X,y) of data and corresponding ground truth
            yield data, target

            # increment for next
            self._curr_idx += 1

    def __len__(self):
        return len(self._df) - self.time_window_size + 1


class BackblazeSingleDrivePtDataset(torch.utils.data.IterableDataset):
    """
    Dataset to iterate over data from a serial number, in chunks of time_window days

    This is an IterableDataset because that makes it easier for chaining.
    If it were a regular Dataset, then random reads would be possible which means
    each csv file could be read multiple times which can be expensive.
    """
    def __init__(self, fpath, feat_cols=None, target_cols=['status'], time_window_size=6, transform=None, target_transform=None):
        super(BackblazeSingleDrivePtDataset, self).__init__()

        # ensure data file exists
        assert os.path.exists(fpath)

        # time frame of data that will be consider one "sequence"
        self.time_window_size = time_window_size

        # current index in iteration
        self._curr_idx = 0

        # load pt as x,y and apply transformations
        self.X, self.y = transform(None), target_transform(None)

        # make sure we have labels for all data
        assert len(self.X)==len(self.y)

    def __iter__(self):
        while self._curr_idx < (len(self.X) - self.time_window_size + 1):
            # return current time window chunk and corresponding ground truth
            yield self.X[self._curr_idx: self._curr_idx + self.time_window_size, ...], \
                    self.y[self._curr_idx: self._curr_idx + self.time_window_size, ...]

            # increment for next
            self._curr_idx += 1

    def __len__(self):
        return len(self.X) - self.time_window_size + 1


if __name__ == "__main__":
    # ##################################### CONVERT CSV TO PT ##################################### #
    # data location
    ROOT_DIR = '/home/kachauha/Downloads/data_Q4_2018_serials'
    META_DIR = ospj(ROOT_DIR, 'meta')

    # same name as original root dir but suffixed with _pt
    SAVE_ROOT_DIR = ROOT_DIR + '_pt'

    # get data from all dirs
    all_ser_files = [f for f in gg(ospj(ROOT_DIR, '**/*.csv')) if os.path.isfile(f)]

    # columns used as features and target
    target_cols = ['status']
    feat_cols = list(pd.read_csv(ospj(META_DIR, 'means.csv'), header=None)[0])

    # for ser_fpath in all_ser_files:
    def save_ser(ser_fpath):
        # decompose file path
        fparts = ser_fpath.split('/')
        subfolder = fparts[-2]
        ser = fparts[-1].split('.')[0]

        # load dataframe
        serdf = pd.read_csv(ser_fpath, usecols = ['date'] + feat_cols + target_cols)
        serdf = serdf.sort_values(by='date', ascending=True)
        serdf = serdf.drop('date', axis=1)

        # convert to tensor and save
        torch.save(obj=(serdf[feat_cols].values, serdf[target_cols].values), \
                    f=ospj(SAVE_ROOT_DIR, subfolder, ser+'.pt'))

    _ = Parallel(n_jobs=-1, prefer='threads')(delayed(save_ser)(ser_fpath) for ser_fpath in all_ser_files)
