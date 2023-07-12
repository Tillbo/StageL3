# Organization
- `data` folder : where datasets must be stored
- `save` folder : where results are stored
- `src` folder : code

# Dependencies

Python **3.11** must be installed.  

## List
- numpy
- POT
- pandas
- torch
- torch_geometric
- networkx
- matplotlib
- rdkit.Chem
- scipy (should be already installed with some of the previous libraries)

## Install

Run  
`pip install -r requirements.txt && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html`  
to install dependencies

# Use
## .config file
Different arguments are present in .config file. DO NOT REMOVE THEM

- XP_NAME : name of the experimentation
- PROCESSES : number of processes to use when computation is parallelized
- PERCENTAGE : percentage of graphs to use during computation
- SEED : random seed
- SMILES : wether to use smile data or not (is False, do not plot anything)
- CONNECTED : wether to keep only the largest connected component or not
- N_PLOT_CLUSTERS : number of graphs per clusters to plot
- N_PLOT_COUNTER : number of counterfactuals to plot (per class)
- DATA_NAME : name of the data file (without .pkl)
- ESPAM_NAME : name of the ESPAM data file (without .pkl)
- ATTENTION_NAME : name of the attention file (without .pkl). Can be None

## Different programs files
### distances.py
Computes the distances matrixes

### counterfactuals.py
Computes the counterfactuals. Distances must be computed first

### clustering.py
Computes clusters. Distances must be computed first