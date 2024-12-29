import torch
import typing
import torch.nn as nn
from GCN import GCN
from SkipGCN import SkipGCN
from DropEdgeGCN import DropEdgeGCN
from SkipDropGCN import SkipDropGCN
def train(
    params: typing.Dict,
    dataset
) -> torch.nn.Module:
  """
    This function trains a node classification model and returns the trained model object.
  """
  # set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load dataset
  data = dataset.data
  data = data.to(device)

  # Update parameters
  params["n_classes"] = dataset.num_classes # number of target classes
  params["input_dim"] = dataset.num_features # size of input features

  # Set a model
  if params['model_name'] == 'GCN':
      model = GCN(
        params["input_dim"],
        params["hid_dim"],
        params["n_classes"],
        params["n_layers"]
        ).to(device)
  elif params['model_name'] == 'SkipGCN':
      model = SkipGCN(
        params["input_dim"],
        params["hid_dim"],
        params["n_classes"],
        params["n_layers"]
      ).to(device)
  elif params['model_name'] == 'DropEdgeGCN':
      model = DropEdgeGCN(
        params["input_dim"],
        params["hid_dim"],
        params["n_classes"],
        params["n_layers"],
        params["dropedge_rate"]
      ).to(device)
  elif params['model_name'] == 'SkipDropGCN':
      model = SkipDropGCN(
        params["input_dim"],
        params["hid_dim"],
        params["n_classes"],
        params["n_layers"],
        params["dropedge_rate"]
      ).to(device)
  else:
      raise NotImplementedError

  model.param_init()
  optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
  loss = nn.CrossEntropyLoss()
  losses = []
  train_accuracies = []
  previous_acc = 0
  print(f"--- Training model {params['model_name']} ---")
  for epoch in range(params["epochs"]):
    model.train()
    optimizer.zero_grad()

    # COMPUTE TRAIN LOSS
    logits = model(data.x, data.edge_index,training=True)
    train_loss = loss(logits[data.train_mask], data.y[data.train_mask])
    losses.append(train_loss.item())

    train_loss.backward()
    optimizer.step()
    # COMPUTE TRAIN ACC
    train_acc = (logits[data.train_mask].max(1)[1] == (data.y[data.train_mask])).float().mean()
    train_accuracies.append(train_acc.item())
    val_acc = evaluate(model, data, data.val_mask)

    # EARLY STOPPING
    patience = patience + 1 if val_acc < previous_acc else 0
    previous_acc = val_acc
    if patience >= params["max_patience"]:
      break

    #PRINT
    if epoch % 10 == 0:
      print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
  return model, val_acc

def evaluate(
    model,
    data,
    mask
):
    model.eval()
    logits = model(data.x, data.edge_index, training=False)
    preds = logits[mask].max(1)[1]
    acc = (preds == data.y[mask]).sum().item() / mask.sum().item()
    return acc