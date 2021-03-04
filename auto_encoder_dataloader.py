import torch.utils.data as data
import pandas as pd
import numpy as np


class DataLoaderGenerator:
    """
    A class for creating data loaders for the network. This allows us to control the amount of memory used by the NN
    at each time, as the dataset is fairly large.
    """

    def __init__(self, filename, chunk_size, total_size, input_dim, batch_size, shuffle):
        """
        @param chunk_size: The amount of events we want the network to load simultaneously to the memory. For obs list
                           simply choose chunk_size = total_size
        """
        self.filename = filename
        self.chunk_size = chunk_size
        self.total_size = total_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = 0

    def reset(self):
        self.idx = 0

    def __next__(self):
        if (self.idx + 1) * self.chunk_size > self.total_size:
            raise StopIteration
        else:
            start = self.idx * self.chunk_size
            stop = (self.idx + 1) * self.chunk_size
            if self.filename.endswith(".h5"):
                x = pd.read_hdf(self.filename, start=start, stop=stop)
                labels = x.rename_axis('ID').values[:, -1]
                x = x.rename_axis('ID').values[:, :self.input_dim]
            elif self.filename.endswith(".npy"):
                x = np.load(self.filename)
                labels = x[start:stop, -1]
                x = x[start:stop, :self.input_dim]  # might not be relevant at the end
            else:
                raise FileNotFoundError
            loader = data.DataLoader(x, batch_size=self.batch_size, shuffle=self.shuffle)
            self.idx += 1
            return loader, labels.tolist()

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.floor(self.total_size / self.chunk_size))