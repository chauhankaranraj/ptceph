import os
import torch
import pandas as pd


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
    def __init__(self, fpath, time_window_size=6, transform=None, target_transform=None):
        super(BackblazeSingleDrivePtDataset, self).__init__()

        # ensure data file exists
        assert os.path.exists(fpath)

        # time frame of data that will be consider one "sequence"
        self.time_window_size = time_window_size

        # current index in iteration
        self._curr_idx = 0

        # load pt as x,y and apply transformations
        self.X, self.y = torch.load(fpath)

        # make sure we have labels for all data
        assert len(self.X)==len(self.y)

        # save transformation functions
        self.transform, self.target_transform = transform, target_transform

    def __iter__(self):
        while self._curr_idx < (len(self.X) - self.time_window_size + 1):
            # return current time window chunk and corresponding ground truth
            yield self.transform(self.X[self._curr_idx: self._curr_idx + self.time_window_size, ...]), \
                    self.target_transform(self.y[self._curr_idx: self._curr_idx + self.time_window_size, ...])

            # increment for next
            self._curr_idx += 1

    def __len__(self):
        return len(self.X) - self.time_window_size + 1


if __name__ == "__main__":
    pass
