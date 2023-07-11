# Organiztion
- `data` folder : where datasets must be stored
- `save` folder : where results are stored
- `src` folder : code

# Dependencies
Python 3.11 must be installed.  
Run  
`pip install -r requirements.txt && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html`  
to install dependencies


Change NMAX to -1 in distances.py to compute more distances
Change NPROCESS in distances.py to use multiple processes

# Dependencies
- numpy
- POT
- pandas
- torch
- torch_geometric
- networkx
- matplotlib
- rdkit.Chem
- scipy (should be already installed with some of the previous libraries)