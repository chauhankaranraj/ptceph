import os
import torch
import pandas as pd

# TODO: STRATEGY:
# 1. make iterable dataset for single serial number
# 2. chain the single serial datasets using chaindataset

# TODO: decide feat cols and label col
class BackblazeSingleDriveDataset(torch.utils.data.IterableDataset):
    def __init__(self, fpath, feat_cols=None, label_cols=['status'], time_window_size=6, transform=None):
        super(BackblazeSingleDriveDataset, self).__init__()

        # ensure data file exists
        assert os.path.exists(fpath)

        # time frame of data that will be consider one "sequence"
        self.time_window_size = time_window_size

        # metadata for processing
        self.feat_cols = feat_cols
        self.label_cols = label_cols

        # since it is a single serial file, it should fit in memory
        # FIXME: this is a hack to sort (assumes date column will be available in data)
        self._df = pd.read_csv(fpath, usecols=['date']+feat_cols+label_cols)
        self._df = self._df.sort_values(by='date', ascending=True)
        self._df = self._df.drop('date', axis=1)

        # current index in iteration
        self._curr_idx = 0

        # preprocessor
        self.transform = transform

    def __iter__(self):
        while self._curr_idx < (len(self._df) - self.time_window_size + 1):
            # get current time chunk, transform if available
            idx_chunk = self._df.iloc[self._curr_idx: self._curr_idx + self.time_window_size, :]
            if self.transform is not None:
                idx_chunk = self.transform(idx_chunk)

            # return a tuple (X,y) of data and corresponding ground truth
            yield idx_chunk.drop(self.label_cols, axis=1), idx_chunk[self.label_cols].iloc[-1]

            # increment for next
            self._curr_idx += 1

    def __len__(self):
        return len(self._df) - self.time_window_size + 1


if __name__ == "__main__":
    # TRAIN_DIR = '/home/kachauha/Downloads/data_Q4_2018_serials'
    train_serials = ['ZA13Q5GK']
    DATA_ROOT_DIR = '/home/kachauha/Downloads/data_Q4_2018_serials/failed'
    use_cols = ['smart_1_raw', 'smart_5_raw', 'status']

    # create by chaining single serial datsets
    train_dataset = torch.utils.data.ChainDataset(
        BackblazeSingleDriveDataset(os.path.join(DATA_ROOT_DIR, serial + '.csv'), feat_cols=use_cols)
        for serial in train_serials)

    # train_dataset = BackblazeSingleDriveDataset(fpath=drive_data_path)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for seq,label in train_dataset:
        print('Label = ', label, '\nData = ')
        print(seq)
