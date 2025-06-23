import torch
from typing import Optional

from torch import arange, Tensor
from torch.nn import Sequential
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GraphEncoder(torch.nn.Module):
    def __init__(self, hidden_channel_dimensions:list, edge_dim:Optional[int]=None, pooling_layer:callable=global_mean_pool):
        super().__init__()
        self.layers = Sequential()
        for i in range(1,len(hidden_channel_dimensions)):
            self.layers.append(
                GATv2Conv(in_channels=hidden_channel_dimensions[i-1], out_channels=hidden_channel_dimensions[i], edge_dim=edge_dim)
            )
        self.node_pooling = pooling_layer
        
    def forward(
        self,
        node_features:Tensor,
        edge_index:Tensor,
        batch:Tensor,
        edge_attr:Optional[Tensor]=None,
        node_subset_indices:Optional[Tensor]=None,
    ):
        node_subset_indices = node_subset_indices if node_subset_indices!=None else arange(node_features.shape[0])
        x = node_features
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr).relu()
        
        # 2. Readout layer
        x = self.node_pooling(x[node_subset_indices,:], batch[node_subset_indices])
        
        return x