import torch
import os
import typing
import torch_geometric
import torch.nn.functional as F
import torch_geometric.datasets as datasets
from GCN import GCN
from DropEdgeGCN import DropEdgeGCN
from SkipDropGCN import SkipDropGCN
from SkipGCN import SkipGCN

from visualize import visualise, dimension_reduction, plot_accuracies, savefigs
from train import train

torch.manual_seed(123) # set seed

dataset = datasets.Planetoid(
    root="./",
    name='CiteSeer',
    split="public",
    transform=torch_geometric.transforms.GCNNorm()
  )
print(dataset.data)

training_params = {
    "lr": 0.005,  # learning rate
    "weight_decay": 0.0005,  # weight_decay
    "epochs": 100,  # number of total training epochs
    "max_patience": 5, # number of k for early stopping
    "hid_dim": 64, # size of hidden features
    "n_layers": None, # number of layers
    "dropedge_rate": 0.3
}

val_accuracies = {}
for name in ["GCN","SkipGCN","DropEdgeGCN","SkipDropGCN"]:
  val_accuracies[name] = []
  trained_models = []
  training_params["model_name"] = name
  for num_layers in [0,1,2,4,10]:
      training_params["n_layers"] = num_layers
      model, val_acc = train(training_params,dataset)
      trained_models.append(model)
      val_accuracies[name].append(val_acc)
      savefigs(dimension_reduction(model,dataset),name,num_layers)
  
  feature_dict = {
    "0_layer": dimension_reduction(trained_models[0],dataset),
    "1_layer": dimension_reduction(trained_models[1],dataset),
    "2_layer": dimension_reduction(trained_models[2],dataset),
    "4_layer": dimension_reduction(trained_models[3],dataset),
    "10_layer": dimension_reduction(trained_models[4],dataset)
  }

  visualise(feature_dict,name)
plot_accuracies(val_accuracies)
print(val_accuracies)

