import numpy as np
import torch

import glob
import numpy as np
np.random.seed(1001)

import torch
from torch_geometric.data import Data, Dataset
from sklearn.datasets import make_blobs


class FakeDataset(Dataset):
    """
    Random number dataset to test with.
    Generates numbers on the fly, but also caches them so .get(i) will return
    something consistent
    """
    def __init__(self, n_events=100):
        super(FakeDataset, self).__init__('nofile')
        self.cache = {}
        self.n_events = n_events

    def get(self, i):
        if i >= self.n_events: raise IndexError
        if i not in self.cache:
            n_hits = np.random.randint(10, 100)
            n_clusters = min(np.random.randint(1, 6), n_hits)
            x = np.random.rand(n_hits, 5)
            y = (np.random.rand(n_hits) * n_clusters).astype(np.int8)
            # Also make a cluster 'truth': energy, boundary_x, boundary_y, pid (4)
            y_cluster = np.random.rand(n_clusters, 4)
            # pid (last column) should be an integer; do 3 particle classes now
            y_cluster[:,-1] = np.floor(y_cluster[:,-1] * 3)
            self.cache[i] = Data(
                x = torch.from_numpy(x).type(torch.float),
                y = torch.from_numpy(y),
                truth_cluster_props = torch.from_numpy(y_cluster)
                )
        return self.cache[i]

    def __len__(self):
        return self.n_events

    def len(self):
        return self.n_events


class BlobsDataset(Dataset):
    """
    Dataset around sklearn.datasets.make_blobs
    """
    
    def __init__(self, n_events=100, seed_offset=0):
        super(BlobsDataset, self).__init__('nofile')
        self.cache = {}
        self.n_events = n_events
        self.cluster_space_dim = 2
        self.seed_offset = seed_offset

    def get(self, i):
        if i >= self.n_events: raise IndexError
        if i not in self.cache:
            n_hits = np.random.randint(50, 70)
            n_clusters = min(np.random.randint(2, 4), n_hits)
            n_bkg = np.random.randint(10, 20)
            # Generate the 'signal'
            X, y = make_blobs(
                n_samples=n_hits,
                centers=n_clusters, n_features=self.cluster_space_dim,
                random_state=i+self.seed_offset
                )
            y += 1 # To reserve index 0 for background
            # Add background
            cluster_space_min = np.min(X, axis=0)
            cluster_space_max = np.max(X, axis=0)
            cluster_space_width = cluster_space_max - cluster_space_min
            X_bkg = cluster_space_min + np.random.rand(n_bkg, self.cluster_space_dim)*cluster_space_width
            y_bkg = np.zeros(n_bkg)
            X = np.concatenate((X,X_bkg))
            y = np.concatenate((y,y_bkg))
            # Calculate geom centers
            truth_cluster_props = np.zeros((n_hits+n_bkg,2))
            for i in range(1,n_clusters+1):
                truth_cluster_props[y==i] = np.mean(X[y==i], axis=0)
            # shuffle
            order = np.random.permutation(n_hits+n_bkg)
            X = X[order]
            y = y[order]
            truth_cluster_props = truth_cluster_props[order]
            self.cache[i] = Data(
                x = torch.from_numpy(X).float(),
                y = torch.from_numpy(y).long(),
                truth_cluster_props = torch.from_numpy(truth_cluster_props).float()
                )
        return self.cache[i]

    def __len__(self):
        return self.n_events

    def len(self):
        return self.n_events
        

class TauDataset(Dataset):
    """Tau dataset.
    
    Features in x:
    recHitEnergy,
    recHitEta,
    zeroFeature, #indicator if it is track or not
    recHitTheta,
    recHitR,
    recHitX,
    recHitY,
    recHitZ,
    recHitTime
    (https://github.com/cms-pepr/HGCalML/blob/master/modules/datastructures/TrainData_NanoML.py#L211-L221)

    Args:
        flip (bool): If True, flips the negative endcap z-values to positive
        reduce_noise (float): Randomly delete a fraction of noise. Useful
            to speed up training.
    """
    def __init__(self, path, flip=True, reduce_noise: float=None):
        super(TauDataset, self).__init__(path)
        self.npzs = list(sorted(glob.iglob(path + '/*.npz')))
        self.flip = flip
        self.reduce_noise = reduce_noise
        self.noise_index = -1
        self.noise_mask_cache = {}

    def blacklist(self, npzs):
        """
        Remove a list of npzs from the dataset
        Useful to remove bad events
        """
        for npz in npzs: self.npzs.remove(npz)

    def get(self, i):
        d = np.load(self.npzs[i])
        x = d['recHitFeatures']
        y = d['recHitTruthClusterIdx'].squeeze()
        if self.flip and np.mean(x[:,7]) < 0:
            # Negative endcap: Flip z-dependent coordinates
            x[:,1] *= -1 # eta
            x[:,7] *= -1 # z
        if self.reduce_noise:
            # Throw away a fraction of noise
            # Have to be careful to throw away to same noise upon
            # future calls of this function.
            mask = self.noise_mask_cache.setdefault(i, mask_fraction_of_noise(y, self.reduce_noise, self.noise_index))
            x = x[mask]
            y = y[mask]
        cluster_index = incremental_cluster_index_np(y.squeeze(), noise_index=self.noise_index)
        if np.all(cluster_index == 0): print('WARNING: No objects in', self.npzs[i])
        truth_cluster_props = np.hstack((
            d['recHitTruthEnergy'],
            d['recHitTruthPosition'],
            d['recHitTruthTime'],
            d['recHitTruthID'],
            ))
        if self.reduce_noise: truth_cluster_props = truth_cluster_props[mask]
        assert truth_cluster_props.shape == (x.shape[0], 5)
        order = cluster_index.argsort()
        return Data(
            x = torch.from_numpy(x[order]).type(torch.float),
            y = torch.from_numpy(cluster_index[order]).type(torch.int),
            truth_cluster_props = torch.from_numpy(truth_cluster_props[order]).type(torch.float),
            inpz = torch.Tensor([i])
            )

    def __len__(self):
        return len(self.npzs)
    def len(self):
        return len(self.npzs)

    def split(self, fraction):
        """
        Creates two new instances of TauDataset with a fraction of events split
        """
        left = self.__class__(self.root, self.flip, self.reduce_noise)
        right = self.__class__(self.root, self.flip, self.reduce_noise)
        split_index = int(fraction*len(self))
        left.npzs = self.npzs[:split_index]
        right.npzs = self.npzs[split_index:]
        return left, right


def incremental_cluster_index(input: torch.Tensor, noise_index=None):
    """
    Build a map that translates arbitrary indices to ordered starting from zero

    By default the first unique index will be 0 in the output, the next 1, etc.
    E.g. [13 -1 -1 13 -1 13 13 42 -1 -1] -> [0 1 1 0 1 0 0 2 1 1]

    If noise_index is not None, the output will be 0 where input==noise_index:
    E.g. noise_index=-1, [13 -1 -1 13 -1 13 13 42 -1 -1] -> [1 0 0 1 0 1 1 2 0 0]

    If noise_index is not None but the input does not contain noise_index, 0
    will still be reserved for it:
    E.g. noise_index=-1, [13 4 4 13 4 13 13 42 4 4] -> [1 2 2 1 2 1 1 3 2 2]
    """
    unique_indices, locations = torch.unique(input, return_inverse=True, sorted=True)
    cluster_index_map = torch.arange(unique_indices.size(0))
    if noise_index is not None:
        if noise_index in unique_indices:
            # Sort so that 0 aligns with the noise_index
            cluster_index_map = cluster_index_map[(unique_indices != noise_index).argsort()]
        else:
            # Still reserve 0 for noise, even if it's not present
            cluster_index_map += 1
    return torch.gather(cluster_index_map, 0, locations).long()


def incremental_cluster_index_np(input: np.array, noise_index=None):
    """
    Reimplementation of incremental_cluster_index for numpy arrays
    """
    unique_indices, locations = np.unique(input, return_inverse=True)
    cluster_index_map = np.arange(unique_indices.shape[0])
    if noise_index is not None:
        if noise_index in unique_indices:
            # Sort so that 0 aligns with the noise_index
            cluster_index_map = cluster_index_map[(unique_indices != noise_index).argsort()]
        else:
            # Still reserve 0 for noise, even if it's not present
            cluster_index_map += 1
    return np.take(cluster_index_map, locations)

def mask_fraction_of_noise(y: np.array, reduce_fraction: float, noise_index: int=-1) -> np.array:
    """Create a mask that throws out a fraction of noise (but keeps all signal)."""
    is_noise = y == noise_index
    n_noise = is_noise.sum()
    n_target_noise = (1.-reduce_fraction) * n_noise
    noise_mask = np.random.permutation(n_noise) < n_target_noise
    mask = np.ones(y.shape[0], dtype=bool)
    mask[is_noise] = noise_mask
    return mask