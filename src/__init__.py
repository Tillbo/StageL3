from .parsing import parse, parse_and_transform, parse_transform_espam
from .const import *
from .draw import plot, save_mol_folder
from .graph import mean_structure, all_to_all
from .transform import transform
from .transport import one_one_parallelised, fgw, node_dists
from .utils import *
from .clustering import cluster_dist, cluster_num, cluster_dbscan, dendo

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from os import mkdir
from shutil import copy

np.set_printoptions(linewidth=400)
np.random.seed(123456789)

#Read configuration file
config = {}
with open(".config", "r") as f:
    s = f.read().split("\n")[:-1]

for l in s:
    kv = l.split(":")
    key = kv[0]
    value = kv[1]
    
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except:
            pass
    if value == 'True':
        value = True
    elif value == 'False':
        value = False
    elif value == 'None':
        value = None
    config[key] = value

try:
    mkdir(f"save/{config['XP_NAME']}")
except FileExistsError:
    pass

copy(".config", f"save/{config['XP_NAME']}")