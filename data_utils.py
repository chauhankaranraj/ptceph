import os
import torch
import pandas as pd

# TODO: STRATEGY:
# 1. make iterable dataset for single serial number
# 2. chain the single serial datasets using chaindataset

# TODO: decide feat cols and label col
class BackblazeSingleDriveDataset(torch.utils.data.IterableDataset):
    def __init__(self, fpath, time_window_size=6, transform=None):
        super(BackblazeSingleDriveDataset, self).__init__()

        # ensure data file exists
        assert os.path.exists(fpath)

        # time frame of data that will be consider one "sequence"
        self.time_window_size = time_window_size

        # since it is a single serial file, it should fit in memory
        self._df = pd.read_csv(fpath)
        self._df = self._df.sort_values(by='date', ascending=True)

        # current index in iteration
        self._curr_idx = 0

        # preprocessor
        self.transform = transform

    def __iter__(self):
        while self._curr_idx < (len(self._df) - self.time_window_size + 1):
            # get current time chunk
            idx_chunk = self._df.iloc[self._curr_idx: self._curr_idx + self.time_window_size, :]
            if self.transform is not None:
                idx_chunk = self.transform(idx_chunk)
            print('---------- yielding index', self._curr_idx, 'chunk ----------')
            yield idx_chunk

            # increment for next
            self._curr_idx += 1

    def __len__(self):
        return len(self._df) - self.time_window_size + 1


# class BackblazeMultipleDrivesDataset(torch.utils.data.ChainDataset):
#     def __init__(self, serial_numbers)


if __name__ == "__main__":
    # TRAIN_DIR = '/home/kachauha/Downloads/data_Q4_2018_serials'
    train_serials = ['2EG2J15G', '2EG3MM0J']
    DATA_ROOT_DIR = '/home/kachauha/Downloads/data_Q4_2018_serials'

    # create by chaining single serial datsets
    train_dataset = torch.utils.data.ChainDataset(
        BackblazeSingleDriveDataset(os.path.join(DATA_ROOT_DIR, serial + '.csv'))
        for serial in train_serials)

    # train_dataset = BackblazeSingleDriveDataset(fpath=drive_data_path)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for seq in train_dataset:
        print(seq)
