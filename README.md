# GRL Mini-project

First, install the dependencies:

python -c "import torch; print(torch.__version__)"

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

pip install git+https://github.com/pyg-team/pytorch_geometric.git
 
To run the experiment, simply run `python experiment.py` or follow the `notebook`.
