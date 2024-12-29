import torch
from torch_geometric.nn import GCNConv, JumpingKnowledge, DeepGCNLayer
import torch.nn as nn
import torch.nn.functional as F

class FrobeniusNormGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(FrobeniusNormGCNConv, self).__init__(in_channels, out_channels, **kwargs)

    def normalize_weights(self):
        # Access the weight matrix from the parameters
        weight = self.lin.weight  # GCNConv uses a linear layer internally
        frobenius_norm = weight.data.norm(p='fro')  # Compute the Frobenius norm
        weight.data /= frobenius_norm  # Normalize the weight matrix

    def forward(self, x, edge_index):
        # Normalize weights before the forward pass
        self.normalize_weights()
        return super(FrobeniusNormGCNConv, self).forward(x, edge_index)

class SkipGCN(nn.Module):
  def __init__(
      self,
      input_dim: int,
      hid_dim: int,
      n_classes: int,
      n_layers: int,
      dropout_ratio: float = 0.3):
    super(SkipGCN, self).__init__()
    """
    Args:
      input_dim: input feature dimension
      hid_dim: hidden feature dimension
      n_classes: number of target classes
      n_layers: number of layers
    """
    self.input_dim = input_dim
    self.hid_dim = hid_dim
    self.n_classes = n_classes
    self.n_layers = n_layers
    self.dropout_ratio = dropout_ratio
    self.layers = nn.ModuleList()
    if self.n_layers == 0:
      self.layers.append(nn.Linear(self.input_dim, self.n_classes))
    else:
      self.layers.append(FrobeniusNormGCNConv(self.input_dim, self.hid_dim))
      for i in range(self.n_layers - 1):
        self.layers.append(FrobeniusNormGCNConv(self.hid_dim, self.hid_dim))
      self.layers.append(nn.Linear(self.hid_dim, self.n_classes))

  def forward(self, X, A,training=True) -> torch.Tensor:
    if self.n_layers > 0:
      X = self.layers[0](X,A)
    skip = X
    for layer in self.layers[1:-1]:
        X = layer(X, A)
        X = F.relu(X)
        X = F.dropout(X, self.dropout_ratio, training=training)

        X = X + skip
        skip = X

    X = self.layers[-1](X)
    return X

  def generate_node_embeddings(self, X, A, training=True) -> torch.Tensor:
    if self.n_layers > 0:
      X = self.layers[0](X,A)
    skip = X
    for layer in self.layers[1:-1]:
      X = layer(X, A)
      X = F.relu(X)
      X = F.dropout(X, self.dropout_ratio, training=training)

      X = X + skip
      skip = X

    return X

  def param_init(self):
    for layer in self.layers:
      layer.reset_parameters()