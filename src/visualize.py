import pandas as pd
import warnings
import matplotlib.pyplot as plt
import torch.nn as nn
import typing
from sklearn.manifold import TSNE
warnings.simplefilter(action='ignore', category=FutureWarning)
def dimension_reduction(model: nn.Module,dataset) -> pd.DataFrame:
  """
    Args:
      model: model object for generating features

    Return:
      pd.DataFrame: A data frame that has 'dimension 1', 'dimension 2', and 'labels' as a column
  """
  tsne = TSNE(n_components=2, random_state=0)
  data = dataset.data
  data = data.to("cpu")
  embeddings = model.generate_node_embeddings(data.x, data.edge_index, training=False)[data.val_mask]
  X = tsne.fit_transform(embeddings.detach().numpy())
  df = pd.DataFrame(X, columns=['dimension 1', 'dimension 2'])
  df['labels'] = data.y[data.val_mask].numpy()
  return df

def visualise(feature_dict: typing.Dict,model_name) -> None:
  fig, axs = plt.subplots(2, 4, figsize=(16, 6))
  if model_name == "GCN":
    axs[1,0].scatter(feature_dict["0_layer"]["dimension 1"], feature_dict["0_layer"]["dimension 2"], c=feature_dict["0_layer"]["labels"])
    axs[1,0].set_title("0_layer")
  axs[0,0].scatter(feature_dict["1_layer"]["dimension 1"], feature_dict["1_layer"]["dimension 2"], c=feature_dict["1_layer"]["labels"])
  axs[0,0].set_title("1_layer")
  axs[0,1].scatter(feature_dict["2_layer"]["dimension 1"], feature_dict["2_layer"]["dimension 2"], c=feature_dict["2_layer"]["labels"])
  axs[0,1].set_title("2_layer")
  axs[0,2].scatter(feature_dict["4_layer"]["dimension 1"], feature_dict["4_layer"]["dimension 2"], c=feature_dict["4_layer"]["labels"])
  axs[0,2].set_title("4_layer")
  axs[0,3].scatter(feature_dict["10_layer"]["dimension 1"], feature_dict["10_layer"]["dimension 2"], c=feature_dict["10_layer"]["labels"])
  axs[0,3].set_title("10_layer")
  plt.savefig(f'./figures/CiteSeer/embeddings_{model_name}')
  plt.show()

def plot_accuracies(val_accuracies):
    """
    Plots the validation accuracies of different GNN models as a function of the number of layers.

    Args:
    val_accuracies (dict): A dictionary where the keys are model names and the values are lists
                            of validation accuracies for 0, 1, 3, 10, and 16 layers.
    """
    # Define the x-axis values representing the number of layers
    layers = [0, 1, 2, 4, 10]

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot each model's accuracies
    for model, accuracies in val_accuracies.items():
        plt.plot(layers, accuracies, marker='o', label=model)
    
    # Adding labels and title
    plt.xlabel('Number of Layers')
    plt.ylabel('Validation Accuracy')
    plt.title('CiteSeer')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/CiteSeer/accuracies')
    plt.show()

def savefigs(feature_dict,name,num_layers):
    plt.figure(figsize=(8, 4))
    plt.scatter(feature_dict["dimension 1"], feature_dict["dimension 2"], c=feature_dict["labels"])
    plt.title(f"{num_layers}_layer")

    plt.savefig(f'figures/CiteSeer/{name}/{num_layers}layers')
