import os
import torch
import pandas as pd


# def bb_data_transform(df):
#     # scale from bytes to gigabyte
#     # FIXME: this is done coz mean centering does not work w/ large numbers
#     df['capacity_bytes'] /= 10**9

#     # get metadata for scaling
#     META_DIR = '/home/kachauha/Downloads/data_Q4_2018_serials/meta'
#     means = pd.read_csv(os.path.join(META_DIR, 'means.csv'), header=None).set_index(0).transpose()
#     stds = pd.read_csv(os.path.join(META_DIR, 'stds.csv'), header=None).set_index(0).transpose()

#     # 0 mean, 1 std
#     # FIXME: subtract and divide w/out using index else nans
#     df = (df - means.values)

#     # FIXME: divide by zero error
#     if (stds.values==0).any():
#         # print('DivideByZeroWarning: std has 0. Dividing will result in nans. Replacing with 1\'s')
#         stds = stds.replace(to_replace=0, value=1)
#     df = df / stds.values

#     # to tensor
#     return torch.Tensor(df.values)



# TODO: decide feat cols and label col
class BackblazeSingleDriveDataset(torch.utils.data.IterableDataset):
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
