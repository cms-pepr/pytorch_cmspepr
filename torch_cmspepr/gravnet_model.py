import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_min, scatter_max, scatter_mean

from torch_cmspepr import GravNetConv
from torch_cmspepr.objectcondensation import scatter_count


def global_exchange(x: Tensor, batch: Tensor) -> Tensor:
    """
    Adds columns for the means, mins, and maxs per feature, per batch.
    Assumes x: (n_hits x n_features), batch: (n_hits),
    and that the batches are sorted!
    """
    n_hits_per_event = scatter_count(batch)
    n_hits, n_features = x.size()
    batch_size = batch.max()+1

    # minmeanmax: (batch_size x 3*n_features)
    meanminmax = torch.cat((
        scatter_mean(x, batch, dim=0),
        scatter_min(x, batch, dim=0)[0],
        scatter_max(x, batch, dim=0)[0]
        ), dim=1)
    assert meanminmax.size() == (batch_size, 3*n_features)

    meanminmax = torch.repeat_interleave(meanminmax, n_hits_per_event, dim=0)
    assert meanminmax.size() == (n_hits, 3*n_features)

    out = torch.cat((meanminmax, x), dim=1)
    assert out.size() == (n_hits, 4*n_features)
    assert out.device == x.device
    return out


# FROM https://link.springer.com/content/pdf/10.1140/epjc/s10052-019-7113-9.pdf:

# GravNet model: The model consists of four blocks. Each
# block starts with concatenating the mean of the vertex
# features to the vertex features, three dense layers with
# 64 nodes and tanh activation, and one GravNet layer
# with S = 4 coordinate dimensions, FLR = 22 features to
# propagate, and FOUT = 48 output nodes per vertex. For
# each vertex, 40 neighbours are considered. The output
# of each block is passed as input to the next block and
# added to a list containing the output of all blocks. This
# determines the full vector of vertex features passed to a
# final dense layer with 128 nodes and ReLU activation

# In all cases, each output vertex of these model building blocks
# is fed through one dense layer with ReLU activation and three
# nodes, followed by a dense layer with two output nodes and
# softmax activation. This last processing step deter- mines the
# energy fraction belonging to each shower. Batch normalisation
# is applied in all models to the input and after each block.

class GravNetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int, out_channels: int = 96,
        space_dimensions: int = 4, propagate_dimensions: int = 22, k: int = 40
        ):
        super(GravNetBlock, self).__init__()
        # Includes all layers up to the global_exchange
        self.gravnet_layer = GravNetConv(
                in_channels, out_channels,
                space_dimensions, propagate_dimensions, k
                )
        self.post_gravnet = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, 128),
            nn.Tanh(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 96),
            nn.Tanh(),
            )
        self.output = nn.Sequential(
            nn.Linear(4*96, 96),
            nn.Tanh(),
            nn.BatchNorm1d(96)
            )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        x = self.gravnet_layer(x, batch)
        x = self.post_gravnet(x)
        assert x.size(1) == 96
        x = global_exchange(x, batch)
        x = self.output(x)
        assert x.size(1) == 96
        return x


class GravnetModel(nn.Module):

    def __init__(
        self, 
        input_dim: int=5,
        output_dim: int=4,
        n_gravnet_blocks: int=4,
        n_postgn_dense_blocks: int=4,
        ):
        super(GravnetModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = n_gravnet_blocks
        self.n_postgn_dense_blocks = n_postgn_dense_blocks

        self.batchnorm1 = nn.BatchNorm1d(self.input_dim)
        self.input = nn.Linear(4*input_dim, 64)

        # Note: out_channels of the internal gravnet layer
        # not clearly specified in paper
        self.gravnet_blocks = nn.ModuleList([
            GravNetBlock(64 if i==0 else 96) for i in range(self.n_gravnet_blocks)
            ])

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend([
                nn.Linear(4*96 if i==0 else 128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                ])
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)
        
        # Output block
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
            )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        device = x.device
        # print('forward called on device', device)
        x = self.batchnorm1(x)
        x = global_exchange(x, batch)
        x = self.input(x)
        assert x.device == device

        x_gravnet_per_block = [] # To store intermediate outputs
        for gravnet_block in self.gravnet_blocks:
            x = gravnet_block(x, batch)
            x_gravnet_per_block.append(x)
        x = torch.cat(x_gravnet_per_block, dim=-1)
        assert x.size() == (x.size(0), 4*96)
        assert x.device == device

        x = self.postgn_dense(x)
        x = self.output(x)
        assert x.device == device
        return x
