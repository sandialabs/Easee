### Imports ###


# Compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)


# Ease of Use
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


# Data and numerical utilities
import numpy as np
import pandas as pd
from collections import Counter


# Network and graphs utilities
import networkx as nx


# Set seeds
import numpy.random
import random
import os

GLOBAL_SEED = 1993
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)


# General utilities
import time
from copy import deepcopy
import os.path
from glob import glob


# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Specific plotting utilities
from matplotlib.patches import Rectangle

# Set plotting font sizes and properties
TINY_SIZE = 24
SMALL_SIZE = 28
MEDIUM_SIZE = 32
BIGGER_SIZE = 36
MARKER_SIZE = 16
LINE_SIZE = 5

plt.rc("font", size = SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize = BIGGER_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize = TINY_SIZE)    # legend fontsize
plt.rc("figure", titlesize = BIGGER_SIZE)  # fontsize of the figure title
plt.rc("lines", markersize = MARKER_SIZE) # marker size
plt.rc("lines", linewidth = LINE_SIZE) # line width


mpl.rcParams["figure.dpi"] = 180

# Height and width per row and column of subplots
FIG_HEIGHT = 9
FIG_WIDTH = 11




### EASEE Wrapper Functions ###


def EASEE_wrapper(DATA_PATH:str, SCRIPT_PATH:str,
                  OUT_PATH:str, out_base:str, 
                  log_file:str):
    """
        EASEE_wrapper(DATA_PATH:str, SCRIPT_PATH:str,
                  OUT_PATH:str, out_base:str, 
                  log_file:str)
        
    # Inputs:
        - DATA_PATH: filename of data file. The format is
            node1,node2,timestamp
        - SCRIPT_PATH: folder containing EASEE script (CreateGraphs.py)
        - OUT_PATH: folder to write EASEE output to
        - out_base: EASEE output filename base
        - log_file: file to capture EASEE stderr output
        
    # Outputs:
        - Nothing is returned
        - EASEE output files are generated. 
        
    # Depends:
        - SCRIPT_PATH/CreateGraphs.py
            CreateGraphs.py's dependencies
        - os
    """
    
    # Clear old output
    os.system("rm " + OUT_PATH + out_base + "*")
    os.system("rm " + OUT_PATH + log_file)
    
    terminal_string = "python " + SCRIPT_PATH + "CreateGraphs.py " + \
        DATA_PATH + " " + OUT_PATH + out_base + \
        " > " + OUT_PATH + log_file
    
    os.system(terminal_string)
    
    
    
def MergeGraphs_wrapper(SCRIPT_PATH:str,
                OUT_PATH:str, in_base:str, 
                out_base:str,
                log_file:str):
    """
        MergeGraphs_wrapper(SCRIPT_PATH:str,
                OUT_PATH:str, in_base:str, 
                out_base:str,
                log_file:str)
        
    # Inputs:
        - SCRIPT_PATH: folder containing EASEE scripts (MergeGraphs.py)
        - OUT_PATH: folder to write output to
        - in_base: EASEE output filename base
        - out_base: MergeGraphs output filename base
        - log_file: file to capture stderr output
        
    # Outputs:
        - Nothing is returned
        - MergeGraphs output files are generated. 
        
    # Depends:
        - SCRIPT_PATH/MergeGraphs.py
            MergeGraphs.py's dependencies
        - os
        - from glob import glob
    """
    
    n_files = len(glob(OUT_PATH + in_base + "*"))
    terminal_string = "python " + SCRIPT_PATH + "MergeGraphs.py " + \
            OUT_PATH + in_base + " " + str(n_files) + \
            " " + OUT_PATH + out_base + \
            " > " + OUT_PATH + log_file
    
    os.system(terminal_string)
    
    
    
def load_EASEE_output(OUT_PATH:str, out_base:str, 
                      timescale, min_comp_size, 
                      DF_DICT_FLAG:bool = False):
    """
        load_EASEE_output(OUT_PATH:str, out_base:str, 
                      timescale, min_comp_size, 
                      DF_DICT_FLAG:bool = False) 
                      
    # Inputs:
        - OUT_PATH: folder to write EASEE output to
        - EASEE output filename base
        - timescale: 1000 for milliseconds, 1 for seconds: how to rescale timestamps to seconds
        - min_comp_size: Minimum connected component size to count
        - DF_DICT_FLAG: if True, return a dictionary of the edge dataframes
    # Outputs:
        - EASEE_windows: Pandas DataFrame with EASEE results
        - if DF_DICT_FLAG: df_dict: keys are snapshot indices, values are dataframes of edges in snapshot
        
    # Depends:
        - pandas as pd
        - networkx as nx
        - numpy as np
    """
    
    # Count the number of EASEE Windows
    n_graphs = len(glob(OUT_PATH + out_base + "*"))
    file_ext = ".csv"
    
    # Loop over snapshots
    quantities = ["snapshot_index", 
                  "unique_nodes", "EA_count", "unique_edges", 
                  "timespan", "components", 
                  "EA_mean", "EA_median", "EA_std"]
    
    EASEE_windows = pd.DataFrame(columns = quantities)
    if DF_DICT_FLAG:
        df_dict = {}
    for ng in range(n_graphs):        
        edge_df = pd.read_csv(OUT_PATH + out_base + str(ng) + file_ext)
        # Extract list of nodes
        node_list = list(edge_df["node_a"]) + list(edge_df["node_b"])
        
        tmp_a = edge_df.groupby("node_a").sum()["num_times"]
        tmp_b = edge_df.groupby("node_b").sum()["num_times"]

        # Compute node appearance statistics
        node_appearances = {}
        for nd in set(node_list):
            if nd in tmp_a and nd in tmp_b:
                node_appearances[nd] = tmp_a[nd] + tmp_b[nd] 
            elif nd in tmp_a:
                node_appearances[nd] = tmp_a[nd] 
            elif nd in tmp_b:
                node_appearances[nd] = tmp_b[nd] 
        
        # Form graph object
        G = nx.from_pandas_edgelist(edge_df[["node_a", "node_b"]], "node_a", "node_b")
        G_comp_sizes = [len(c) for c in sorted(nx.connected_components(G), key = len, reverse = True)]
        G_comp_sizes = [x for x in G_comp_sizes if x >= min_comp_size]
        
        # Append 
        tmp = {"snapshot_index" : ng,
            "unique_nodes" : len(set(node_list)),
            "EA_count" : np.sum(edge_df["num_times"]),
            "unique_edges" : edge_df.shape[0], 
            "timespan" : (edge_df["last_time"].max() - edge_df["first_time"].min()) / timescale,
            "components" : len(G_comp_sizes),
            "EA_mean" : np.mean(edge_df["num_times"]),
            "EA_median" : np.median(edge_df["num_times"]),
            "EA_std" : np.std(edge_df["num_times"]),
            "NA_mean" : np.mean(list(node_appearances.values())),
            "NA_median" : np.median(list(node_appearances.values())),
            "NA_std" : np.std(list(node_appearances.values()))}
        
        EASEE_windows = EASEE_windows.append(tmp, ignore_index = True)
        if DF_DICT_FLAG:
            df_dict[ng] = deepcopy(edge_df)
        
    # Various density measures
    EASEE_windows["unique_density"] = EASEE_windows["unique_edges"] / EASEE_windows["unique_nodes"]
    EASEE_windows["EA_density"] = EASEE_windows["EA_count"] / EASEE_windows["unique_nodes"]
    EASEE_windows["EA_ratio"] = EASEE_windows["EA_count"] / EASEE_windows["unique_edges"]
    EASEE_windows["possible_density"] = 2 * EASEE_windows["unique_edges"] / (EASEE_windows["unique_nodes"] * (EASEE_windows["unique_nodes"] - 1))
    
    if DF_DICT_FLAG:
        return EASEE_windows, df_dict
    else: 
        return EASEE_windows

    
    
### Other Useful Functions ###

"""
    rice_bins(x)
    
# Inputs: 
    - Vector x
    
# Outputs:
    - Logarithmically spaced bins for a histogram (using Rice's Rule)
    
# Depends:
    - numpy as np
"""
rice_bins = lambda x : np.unique(np.round(np.logspace(0.0, 
                        np.log10(np.max(x)), 
                        2 * int(np.ceil(np.power(len(x), 1.0 / 3.0))))))



def find_merge_groups(sim_mat:np.array, thresh:float):
    """
       find_merge_groups(sim_mat:np.array, thresh:float)
    
    # Inputs:
        - Square, symmetric similarity matrix: numpy array
        - threshold: float: for deciding whether two groups/entries are close enough
        
    # Outputs:
        - group_list: list of lists of groups to merge
        
    # Depends:
        - numpy as np
        
    # Notes:
        Works with the loaded output of MergeGraphs
    """
    assert 2 == len(sim_mat.shape), "Wrong number of dimensions"
    assert sim_mat.shape[0] == sim_mat.shape[1], "Matrix is not square"
    assert np.allclose(sim_mat, sim_mat.T), "Matrix is not symmetric"
    
    # Apply threshold
    sim_mat = (sim_mat >= thresh)    
    
    group_list = []
    group_start_idx = 0
    for idx in range(1, sim_mat.shape[0]):
        if 1.0 > np.mean(sim_mat[np.ix_(range(group_start_idx, idx + 1), 
                                       range(group_start_idx, idx + 1))]):
            group_list += [list(range(group_start_idx, idx))]
            group_start_idx = idx
    group_list += [list(range(group_start_idx, idx + 1))]
    
    return group_list



