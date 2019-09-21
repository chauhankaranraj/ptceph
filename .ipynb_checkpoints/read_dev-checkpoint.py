import torch
from petastorm import make_reader, make_batch_reader
from petastorm.pytorch import DataLoader

DATA_DIR = 'file:///home/kachauha/Downloads/data_Q4_2018_parquet/part.0.parquet'
with DataLoader(make_batch_reader(DATA_DIR, num_epochs=10), batch_size=64) as train_loader:
    for batch in train_loader:
        print(batch)
