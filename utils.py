import os
from glob import glob as gg
from os.path import join as ospj
from itertools import chain, cycle
from joblib import Parallel, delayed

import torch
import numpy as np
import scipy as sp
import pandas as pd
import dask.dataframe as dd

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans


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


def append_rul_days_column(drive_data):
    """Appends remaining-useful-life column to pandas/dask dataframe
    RUL is calculated in terms of days. It is the difference between the date
    on which a disk failed (or if it hasnt failed, then max date of observation)
    and the current row (entry in time series) date

    Arguments:
        drive_data {dataframe.groupby.group} -- group of a given hard drive

    Returns:
        dask.dataframe/pandas.DataFrame -- dataframe with the added column
    """
    return drive_data.assign(rul_days=drive_data['date'].max()-drive_data['date'])


def rolling_featurize(df, window=6, drop_cols=('date', 'failure', 'capacity_bytes', 'rul'), group_cols=('serial_number'), cap=True):
    """
    Extracts 6-day rolling features (6-day mean, std, coefficient of variation) from raw data
    in a pandas dataframe
    """
    # save the status labels
    # FIXME: if this is not a df, then earlier versions of pandas (0.19) complains
    statuses = df[['status']]

    # group by serials, drop cols which are not to be aggregated
    if drop_cols is not None:
        grouped_df = df.drop(drop_cols, axis=1).groupby(group_cols)
    else:
        grouped_df = df.groupby(group_cols)

    # feature columns
    featcols = grouped_df.first().columns

    # get mean value in last 6 days
    means = grouped_df.rolling(window)[featcols].mean()

    # get std in last 6 days
    stds = grouped_df.rolling(window)[featcols].std()

    # coefficient of variation
    cvs = stds.divide(means, fill_value=0)

    # rename before mergeing
    means = means.rename(columns={col: 'mean_' + col for col in means.columns})
    stds = stds.rename(columns={col: 'std_' + col for col in stds.columns})
    cvs = cvs.rename(columns={col: 'cv_' + col for col in cvs.columns})

    # combine features into one df
    res = means.merge(stds, left_index=True, right_index=True)
    res = res.merge(cvs, left_index=True, right_index=True)

    # drop rows where all columns are nans
    res = res.dropna(how='all')

    # fill nans created by cv calculation
    res = res.fillna(0)

    # capacity of hard drive
    if cap:
        capacities = df[['serial_number', 'capacity_bytes']].groupby('serial_number').max()
        res = res.merge(capacities, left_index=True, right_index=True)

    # bring serial number back as a col instead of index, preserve the corresponding indices
    res = res.reset_index(level=[0])

    # add status labels back.
    res = res.merge(statuses, left_index=True, right_index=True)

    return res


def get_drive_data_from_json(fnames, serial_numbers):
    # get data for only one failed and one working serial number from the last three days
    subdfs = []
    for fname in fnames:
        # read in raw json
        df = pd.read_json(fname, lines=True)
        # convert to df format to index for serial number. then append to list of sub-dfs
        subdfs.append(df[df['smartctl_json'].apply(pd.Series)['serial_number'].isin(serial_numbers)])

    # merge all sub-dfs into one
    return pd.concat(subdfs, ignore_index=True)


def get_downsampled_working_sers(df, num_serials=300, model=None, scaler=None):
    """Downsample the input dataframe of working hard drives by selecting best
    representatives of the data using clustering. Return the identifiers
    (serial numbers) of these best representative hard drives (cluster centers)

    Arguments:
        df {pd.DataFrame} -- dataframe where each row is the feature vector of
                                a given data point (hard drive)

    Keyword Arguments:
        num_serials {int} -- number of hard drives to keep (default: {300})
        model {sklearn.cluster.KMeans} -- clustering model to be used for
                                finding best hard drives (default: {None})
        scaler {sklearn.preprocessing.RobustScaler} -- scaler to scale the
                                            raw input data (default: {None})

    Returns:
        list -- serial numbers of cluster centers (best repr hard drives)
    """
    # default to robust scaler
    if scaler is None:
        scaler = RobustScaler()

    # default to vanilla kmeans
    if model is None:
        model = KMeans(n_clusters=num_serials,
                    max_iter=1e6,
                    n_jobs=-1)

    # fit model to scaled data
    model.fit(scaler.fit_transform(df))

    # iterate over centers to find the serials that were closest to each center
    working_best_serials = []

    # if model was not dask, dd.compute returns tuple of len 1
    cluster_centers = dd.compute(model.cluster_centers_)
    if isinstance(cluster_centers, tuple):
        cluster_centers = cluster_centers[0]

    for i, c in enumerate(cluster_centers):
        # all the points that belong to this cluster
        cluster_pts = dd.compute(df.iloc[model.labels_==i])
        if isinstance(cluster_pts, tuple):
            cluster_pts = cluster_pts[0]

        # distance of each point to the center
        min_dist_idx = np.argmin(sp.spatial.distance.cdist(cluster_pts, c.reshape(1, -1), metric='euclidean'))
        working_best_serials.append(cluster_pts.iloc[min_dist_idx].name)

    return working_best_serials


def get_train_test_serials(work_dir, fail_dir, test_size=None, train_size=None, oversample=True):
    """Splits serial numbers into train set and test set

    Arguments:
        work_dir {str} -- path to dir where working serials csv's are stored
        fail_dir {str} -- path to dir where failed serials csv's are stored

    Keyword Arguments:
        test_size {float} -- fraction of serials to use for testing. At least one of
                             train_size or test_size must be specified (default: {None})
        train_size {float} -- fraction of serials to use for training. At least one of
                             train_size or test_size must be specified (default: {None})
        oversample {bool} -- if True, chains non-abundant-class serials in cycles till
                             the length of lists of both class serials is equal (default: {True})

    Raises:
        RuntimeError: if neither train_size nor test_size is specified

    Returns:
        (list, list) -- tuple of train_serials list, test_serials list
    """
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

    if oversample:
        # oversampling in train set to relax class imbalance somewhat
        if len(working_files_train) > len(failed_files_train):
            train_ser_files = list(chain(*zip(cycle(failed_files_train), working_files_train)))
        else:
            train_ser_files = list(chain(*zip(cycle(working_files_train), failed_files_train)))
    else:
        train_ser_files = working_files_train + failed_files_train

    # dont oversample in test set - this will skew evaluation results
    test_ser_files = working_files_test + failed_files_test

    return train_ser_files, test_ser_files


def serials_dataset_csv2pt(root_dir=None, meta_dir=None, save_root_dir=None):
    """Converts Backblaze dataset stored by serials (one csv per serial) from
    csv format to pt format for better read performance with pytorch

    Assumes this structure of serials dataset
    -- data_root_dir
      |-- failed
      |---- ST03523.csv, etc
      |-- working
      |---- WR1165.csv, etc

    Assumes this structure of serials dataset metadata
    -- meta_dir
      |-- means.csv
      |-- stds.csv

    NOTE: meta_dir can be inside data_root_dir too

    Keyword Arguments:
        root_dir {str} -- path to serials dataset (default: {None})
        meta_dir {str} -- path to metadata of serials dataset (default: {None})
        save_root_dir {str} -- path where pt format serials dataset is to be stored (default: {None})
    """
    # data location
    if root_dir is None:
        root_dir = '/home/kachauha/Downloads/data_Q4_2018_serials'

    # means, stds of data (for normalization)
    if meta_dir is None:
        meta_dir = ospj(root_dir, 'meta')

    # dir where the '.pt' dataset is to be stored
    # same name as root dir but suffixed with _pt
    if save_root_dir is None:
        save_root_dir = root_dir + '_pt'

    # get data from all dirs
    all_ser_files = [f for f in gg(ospj(root_dir, '**/*.csv')) if os.path.isfile(f)]

    # columns used as features and target
    target_cols = ['status']
    feat_cols = list(pd.read_csv(ospj(meta_dir, 'means.csv'), header=None)[0])

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
        # FIXME: save as torch tensor and not numpy array
        # FIXME: dont save columns that have 0 std
        torch.save(obj=(serdf[feat_cols].values, serdf[target_cols].values), \
                    f=ospj(save_root_dir, subfolder, ser+'.pt'))

    def save_meta(ser_fpath):
        # decompose file path
        fparts = ser_fpath.split('/')
        subfolder = fparts[-2]
        ser = fparts[-1].split('.')[0]

        serdf = pd.read_csv(ser_fpath, header=None).set_index(0).transpose()

        # convert to tensor and save
        torch.save(obj=serdf.values, f=ospj(save_root_dir, subfolder, ser+'.pt'))

    # for saving failed and working drives
    _ = Parallel(n_jobs=-1, prefer='threads')(delayed(save_ser)(ser_fpath) for ser_fpath in all_ser_files)

    # for saving mean, std
    meta_fpaths = [i for i in all_ser_files if 'meta' in i]
    _ = Parallel(n_jobs=-1, prefer='threads')(delayed(save_meta)(ser_fpath) for ser_fpath in meta_fpaths)


def get_nan_count_percent(df, divisor=None):
    """Calculates the number of nan values per column,
        both as an absolute amount and as a percentage of some pre-defined "total" amount

    Arguments:
        df {pandas.DataFrame/dask.dataframe} -- dataframe whose nan count to generate

    Keyword Arguments:
        divisor {int/float} -- the "total" amount for calculating percentage.
                                If value in count column is n, value in percent column
                                will be n/divisor.
                                If not provided, number of rows is used by default
                                (default: {None})

    Returns:
        ret_df {pandas.DataFrame/dask.dataframe} -- dataframe with counts and percentages
                                                    of nans in each column of input df.
                                                    Column name is the index, "count" and
                                                    "percent" are the two columns.
    """
    # if total count is not provided, use the number of rows
    if divisor is None:
        # NOTE: len must be used, not shape because in case of dask dataframe
        # shape returns a delayed computation, not an actual value. but
        # len returns an actual value
        divisor = len(df)

    # get count and convert series to dataframe
    ret_df = df.isna().sum().to_frame("count")

    # add percent column
    ret_df["percent"] = ret_df["count"] / divisor

    return ret_df


def optimal_repartition_df(df, partition_size_bytes=None):
    # ideal partition size as recommended in dask docs
    if partition_size_bytes is None:
        partition_size_bytes = 100 * 10**6

    # determine number of partitions
    df_size_bytes = df.memory_usage(deep=True).sum().compute()
    num_partitions = int(np.ceil(df_size_bytes / partition_size_bytes))

    # repartition
    return df.repartition(npartitions=num_partitions)
