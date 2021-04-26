"""
    SBM_Community_v2.py
    
    Arvind Prasadan, Jeremy D. Wendt, Rich V. Field
    
This script runs a simulation to test the ability of EASEE,
relative to a sliding window, to detect a changepoint in a network with
overlapping community structure. 

Dependencies: 
    - SbmCommunityTestData.py
    - CreateGraphs.py
        - Associated dependencies of CreateGraphs.py
        
Usage:
    python3 SBM_Community_v2 $run_idx
    run_idx is between 0 and 5
"""


# General numerical utilities
import numpy as np
from scipy.optimize import minimize

# Network functionality
import networkx as nx
import community as community_louvain

# General utilities
from random import choice, shuffle
from collections import deque
from copy import deepcopy

# File IO
import sys
import os
import pickle
from glob import glob

# Evaluation of community detection
from sklearn.metrics import adjusted_mutual_info_score

# Local Modules
import SbmCommunityTestData # Network functionality

# Visualization
import matplotlib.pyplot as plt

# Set plotting font sizes and properties
TINY_SIZE = 18
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30
MARKER_SIZE = 14
LINE_SIZE = 4

plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = TINY_SIZE)    # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', markersize = MARKER_SIZE) # marker size
plt.rc('lines', linewidth = LINE_SIZE) # line width



def log_dist_EA(n_edges: int, n_EA: int) -> np.array:
    """
        log_dist_EA(n_edges: int, n_EA: int) -> np.array
    
    # Inputs:
        - n_edges: Number of Unique Edges
        - n_EA: Number of edge advertisements
        
    # Outputs:
        - Distribution of EA counts
        
    This function creates a log-linear distribution of edges. 
    That is, given n_edges edges and a total edge appearance count of n_EA,
    it distributes the edge counts so that the distribution function is log-linear.
    The constraint is that the smallest count is 1.
    """
    
    assert n_EA >= n_edges, "n_EA must be larger than n_edges"
    
    # Internal function to optimize: Squared Deviation
    def opt_fcn(a):
        tmp = np.power(np.arange(1, n_edges + 1), -a)
        return np.abs(np.sum(np.floor(np.power(n_edges, a) * tmp)) - n_EA)**2
    
    # Manual line search to prime optimization
    a_arr = np.arange(-10, 10, 0.001)
    a_opt = a_arr[np.argmin([opt_fcn(x) for x in a_arr])]
    a_opt = minimize(opt_fcn, a_opt).x[0]

    # Actual fit
    dist_edges = np.floor(np.power(n_edges, a_opt) * np.power(np.arange(1, n_edges + 1), -a_opt))
    dist_edges[dist_edges <= 0] = 1
    rem = int(n_EA - np.sum(dist_edges))
    
    # We're not going to be exact, so make a few minor adjustments 
    if 0 == rem:
        return dist_edges
    else:
        while 0 != rem:
            print(rem)
            
            if 0 < rem:
                dist_edges[choice(range(n_edges))] += 1
            elif 0 > rem:
                dist_edges[choice(range(np.where(dist_edges <= 1)[0][0]))] -= 1
            rem = int(n_EA - np.sum(dist_edges))
            
    dist_edges = np.sort(dist_edges).astype(int)[::-1]
    
    assert 0 == np.sum(0 == dist_edges), str(np.sum(0 == dist_edges)) + ", Function failed to assign non-zero EAs to all edges"
    assert np.sum(dist_edges) == n_EA, str(np.sum(dist_edges)) + ", Function failed to assign correct number of EAs"
    return dist_edges



def EA_sequence(edges, dist_edges):
    """
    # Inputs:
        - edges: List of tuples 
        - dist_edges: output of log_dist_EA
    
    # Outputs:
        - list of Edge Advertisements (EAs)
        
    Given a distribution of edge counts and a list of edges, a list of 
    edges is produced (edges appear a specified number of times in random order)
    """
    
    assert len(edges) == len(dist_edges), "inputs are not compatible"
    
    shuffle(dist_edges)
    
    EA = deque()
    for idx in range(len(dist_edges)):
        for x in range(int(dist_edges[idx])):
            EA.append(edges[idx]) 
    EA = list(EA)
    shuffle(EA)
    
    return EA


# Actual script
if "__main__" == __name__:
    
    if 1 < len(sys.argv):
        run_idx = int(sys.argv[1])
    else:
        run_idx = 0
    print(run_idx)
        
    # Parameters

    n_nodes = 2000 # Nodes per graph
    n_EA = 50000 # EAs per graph
    n_blocks = 10 # Blocks per graph
    min_degree = 3 # Lowest degree of a node
    
    # Subgraphs (by EA index) to consider
    starting_points = np.arange(0, n_EA + 1, n_EA / 5).astype(int)
    graph_width = n_EA
    
    p_in_left = 0.05 # Intra-block edge probability
    p_out_left = 0.005 # Inter-block edge probability
    p_in_right = 0.04 # Intra-block edge probability
    p_out_right = 0.005 # Inter-block edge probability
    
    EA_lambda = 25.0 # For timestamps
    
    n_louvain = 10 # How many times to run community detection
    
    # Fixed, disjoint window sizes
    window_sizes = (np.round(np.logspace(3, np.log10(n_EA), 10) / 50) * 50).astype(int)
    
    # Output files for CreateGraphs
    filepath = "./data_" + "p" + str(p_in_left) + "q" + str(p_out_left) + "p" + str(p_in_right) + "q" + str(p_out_right) + "/"
    filebase = "SBM_run" + str(run_idx)  
    fileext = ".csv"

    if not os.path.exists(filepath):
        os.makedirs(filepath)
        
    outdir = "./out/"
    outfile = "run" + str(run_idx) + "p" + str(p_in_left) + "q" + str(p_out_left) + "p" + str(p_in_right) + "q" + str(p_out_right)
    outext = ".pickle"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    
    # Create networks
    print("Creating Networks")
    left_graph = SbmCommunityTestData.min_degree_sbm(n_nodes, n_blocks, 
                                                     p_in_left, p_out_left, min_degree)
    correct_left = SbmCommunityTestData.partition_ids_sbm(n_nodes, n_blocks)
    
    right_graph, mapping = SbmCommunityTestData.reorder_nodes(
        SbmCommunityTestData.min_degree_sbm(n_nodes, n_blocks, 
                                            p_in_right, p_out_right, min_degree))
    correct_right = SbmCommunityTestData.partition_ids_sbm(n_nodes, n_blocks, mapping)
    
    
    # Create Graph from EAs
    print("Creating EAs")
    left_edges = list(set([(min(x), max(x)) for x in left_graph.edges]))
    right_edges = list(set([(min(x), max(x)) for x in right_graph.edges]))
    
    dist_edges = log_dist_EA(len(left_edges), n_EA)
    left_EA = EA_sequence(left_edges, dist_edges)
    dist_edges = log_dist_EA(len(right_edges), n_EA)
    right_EA = EA_sequence(right_edges, dist_edges)
    
    EA_total = left_EA + right_EA

    # Edge Advertisement times
    EA_times = np.cumsum(np.random.exponential(1.0 / EA_lambda, size = n_EA + n_EA))
    split_time = EA_times[n_EA] # When left ends and right begins
    
    
    # Run EASEE on the network
        
    # Write graph to file
    filemid = "TOTAL"
    with open(filepath + filebase + filemid + fileext, "w") as f:
        for idx in range(len(EA_times)):
            f.write(str(EA_total[idx][0]) + "," + str(EA_total[idx][1]) + "," + str(EA_times[idx]) + "\n")
       
   
    
    # Run CreateGraphs
    os.system("python CreateGraphs.py " + filepath + filebase + filemid + fileext 
              + " " + filepath + filebase + filemid 
              + " > " + filepath + filebase + filemid + "_stdout.txt")
    n_CG = len(glob(filepath + filebase + filemid + "_graph_*")) # Number of outputs 
    
    # See where network splits
    time_splits = np.zeros((n_CG + 1, ), dtype = float)
    time_splits[0] = 0
    for nc in range(n_CG):
        time_splits[nc + 1] = np.genfromtxt(filepath + filebase + filemid + "_graph_" + str(nc) + ".csv", delimiter = ",")[:, [2, 3]][1:, ].max().max()
    CG_range_total = np.empty((n_CG, ), dtype = object)
    for nc in range(1, n_CG + 1): # Loop over CreateGraphs segements
        # Form graph object
        idx = np.where((EA_times > time_splits[nc - 1])
                                            & (EA_times <= time_splits[nc]))[0].astype(int)
        CG_range_total[nc - 1] = (idx[0], idx[-1])
    print("CG Total")
    print(CG_range_total)
    
    # Perform community detection on CreateGraphs/EASEE output
    AMI_CG_inner = np.zeros((n_CG, n_louvain))
    for nc in range(1, n_CG + 1): # Loop over CreateGraphs segements
        # Form graph object
        G_inner = nx.Graph([EA_total[x] for x in np.where((EA_times > time_splits[nc - 1]) 
                                            & (EA_times <= time_splits[nc]))[0].astype(int)])
        # Subset to extant nodes
        part_nodes = [x for x in range(n_nodes) if x in G_inner.nodes]
        correct_part_left = [correct_left[x] for x in part_nodes]
        correct_part_right = [correct_right[x] for x in part_nodes]
        
        if time_splits[nc] <= split_time: # Left
            lr_alpha = 1.0
        # Straddles both
        elif (time_splits[nc] > split_time) and (time_splits[nc - 1] <= split_time):
            lr_alpha = np.mean(EA_times[(EA_times > time_splits[nc - 1]) & (EA_times <= time_splits[nc])] <= split_time)
        else:
            lr_alpha = 0.0
        # Perform community detection
        for nl in range(n_louvain):
            partition = community_louvain.best_partition(G_inner)
            partition = [partition[x] for x in part_nodes]
            AMI_CG_inner[nc - 1, nl] = (lr_alpha * adjusted_mutual_info_score(correct_part_left, partition) + 
                    (1.0 - lr_alpha) * adjusted_mutual_info_score(correct_part_right, partition))
    AMI_CG_total = deepcopy(AMI_CG_inner)  
    print("CG Total")
    print(AMI_CG_inner)
    
    
    
    AMI_oracle = np.zeros((len(starting_points), n_louvain, 2))
    AMI_unsubset = np.zeros((len(starting_points), n_louvain))
    AMI_unsubset_unrolled = np.zeros((len(starting_points), n_louvain, 2))
    AMI_CG = np.empty((len(starting_points), ), dtype = object)
    AMI_windows = np.empty((len(starting_points), len(window_sizes)), dtype = list)
    
    start_idx = int(run_idx % len(starting_points))
    starting_point = starting_points[start_idx]
    print(start_idx, starting_point, len(starting_points))
    
    filemid = "_" + str(starting_point) 
    
    # Make graph from subset
    indices = range(starting_point, starting_point + graph_width)
    G_subset = nx.Graph([EA_total[x] for x in indices])
    
    # Find true left/right subsets
    G_subset_left = nx.Graph([EA_total[x] for x in indices if x <= n_EA])
    G_subset_right = nx.Graph([EA_total[x] for x in indices if x > n_EA])
    
    
    # Oracle community detection (known truth)
    
    # Subset to extant nodes
    part_nodes_left = [x for x in range(n_nodes) if x in G_subset_left.nodes]
    correct_part_left = [correct_left[x] for x in part_nodes_left]
    
    part_nodes_right = [x for x in range(n_nodes) if x in G_subset_right.nodes]
    correct_part_right = [correct_right[x] for x in part_nodes_right]

    # Perform oracle community detection
    for nl in range(n_louvain):
        if len(part_nodes_left) > 0:
            partition = community_louvain.best_partition(G_subset_left)
            partition = [partition[x] for x in part_nodes_left]
            AMI_oracle[start_idx, nl, 0] = adjusted_mutual_info_score(correct_part_left, partition)
        else:
            AMI_oracle[start_idx, nl, 0] = np.nan
        if len(part_nodes_right) > 0:
            partition = community_louvain.best_partition(G_subset_right)
            partition = [partition[x] for x in part_nodes_right]
            AMI_oracle[start_idx, nl, 1] = adjusted_mutual_info_score(correct_part_right, partition)
        else:
            AMI_oracle[start_idx, nl, 1] = np.nan
    print("Oracle")
    print(AMI_oracle[start_idx, :, :])    
            
    # Un-subsetted community detection
    # Subset to extant nodes
    part_nodes = [x for x in range(n_nodes) if x in G_subset.nodes]
    correct_part_left = [correct_left[x] for x in part_nodes]
    correct_part_right = [correct_right[x] for x in part_nodes]
    
    lr_alpha = np.mean([x < n_EA for x in indices])
    print("alpha", starting_point, lr_alpha)
    # Perform community detection
    for nl in range(n_louvain):
        partition = community_louvain.best_partition(G_subset)
        partition = [partition[x] for x in part_nodes]
        AMI_unsubset_unrolled[start_idx, nl, 0] = adjusted_mutual_info_score(correct_part_left, partition) 
        AMI_unsubset_unrolled[start_idx, nl, 1] = adjusted_mutual_info_score(correct_part_right, partition)
        AMI_unsubset[start_idx, nl] = (lr_alpha * AMI_unsubset_unrolled[start_idx, nl, 0] + 
                (1.0 - lr_alpha) * AMI_unsubset_unrolled[start_idx, nl, 1])
    print("Unsubset")
    print(AMI_unsubset[start_idx, :])
    print(AMI_unsubset_unrolled[start_idx, :, :])
    
    # EASEE on unsubsetted/unsplit network
        
    # Write graph to file
    with open(filepath + filebase + filemid + fileext, "w") as f:
        for idx in indices:
            f.write(str(EA_total[idx][0]) + "," + str(EA_total[idx][1]) + "," + str(EA_times[idx]) + "\n")
       
   
    
    # Run CreateGraphs
    os.system("python CreateGraphs.py " + filepath + filebase + filemid + fileext 
              + " " + filepath + filebase + filemid 
              + " > " + filepath + filebase + filemid + "_stdout.txt")
    n_CG = len(glob(filepath + filebase + filemid + "_graph_*")) # Number of outputs 
    
    # See where network splits
    time_splits = np.zeros((n_CG + 1, ), dtype = float)
    time_splits[0] = EA_times[starting_point]
    for nc in range(n_CG):
        time_splits[nc + 1] = np.genfromtxt(filepath + filebase + filemid + "_graph_" + str(nc) + ".csv", delimiter = ",")[:, [2, 3]][1:, ].max().max()
    CG_range = np.empty((n_CG, ), dtype = object)
    for nc in range(1, n_CG + 1): # Loop over CreateGraphs segements
        # Form graph object
        idx = np.where((EA_times > time_splits[nc - 1])
                                            & (EA_times <= time_splits[nc]))[0].astype(int)
        CG_range[nc - 1] = (idx[0], idx[-1])
    print("CG")
    print(CG_range)
    
    # Perform community detection on CreateGraphs/EASEE output
    AMI_CG_inner = np.zeros((n_CG, n_louvain))
    for nc in range(1, n_CG + 1): # Loop over CreateGraphs segements
        # Form graph object
        G_inner = nx.Graph([EA_total[x] for x in np.where((EA_times > time_splits[nc - 1]) 
                                            & (EA_times <= time_splits[nc]))[0].astype(int)])
        # Subset to extant nodes
        part_nodes = [x for x in range(n_nodes) if x in G_inner.nodes]
        correct_part_left = [correct_left[x] for x in part_nodes]
        correct_part_right = [correct_right[x] for x in part_nodes]
        
        if time_splits[nc] <= split_time: # Left
            lr_alpha = 1.0
        # Straddles both
        elif (time_splits[nc] > split_time) and (time_splits[nc - 1] <= split_time):
            lr_alpha = np.mean(EA_times[(EA_times > time_splits[nc - 1]) & (EA_times <= time_splits[nc])] <= split_time)
        else:
            lr_alpha = 0.0
        # Perform community detection
        for nl in range(n_louvain):
            partition = community_louvain.best_partition(G_inner)
            partition = [partition[x] for x in part_nodes]
            AMI_CG_inner[nc - 1, nl] = (lr_alpha * adjusted_mutual_info_score(correct_part_left, partition) + 
                    (1.0 - lr_alpha) * adjusted_mutual_info_score(correct_part_right, partition))
    AMI_CG[start_idx] = deepcopy(AMI_CG_inner)  
    print("CG")
    print(AMI_CG_inner)
    
                
    # Fixed window sweep (disjoint)
    # that is, instead of EASEE's intelligent divisions, we use
    # fixed window sizes 
    
    for w_idx, window_size in enumerate(window_sizes):
        print(w_idx, window_size, len(window_sizes))
        
        ami_window = deque()
        for w_start in range(indices[0], indices[-1], window_size):
            # Form Graph from window
            w_end = min(len(EA_total), w_start + window_size)
            G_inner = nx.Graph([EA_total[x] for x in range(w_start, w_end)])
            
            if EA_times[w_end - 1] <= split_time: # Left
                lr_alpha = 1.0
            # Straddles both
            elif (EA_times[w_end - 1] > split_time) and (EA_times[w_start] <= split_time):
                lr_alpha = np.mean(EA_times[(EA_times > EA_times[w_start]) & (EA_times <= EA_times[w_end - 1])] <= split_time)
            else:
                lr_alpha = 0.0
            
            # Subset to extant nodes
            part_nodes = [x for x in range(n_nodes) if x in G_inner.nodes]
            correct_part_left = [correct_left[x] for x in part_nodes]
            correct_part_right = [correct_right[x] for x in part_nodes]
            
            ami_inner = np.zeros((n_louvain, ))
            # Perform community detection
            for nl in range(n_louvain):
                partition = community_louvain.best_partition(G_inner)
                partition = [partition[x] for x in part_nodes]
                ami_inner[nl] = (lr_alpha * adjusted_mutual_info_score(correct_part_left, partition) + 
                        (1.0 - lr_alpha) * adjusted_mutual_info_score(correct_part_right, partition))
            ami_window.append(ami_inner)
        AMI_windows[start_idx, w_idx] = deepcopy(list(ami_window))
        print(ami_window)
    
           
            
    # Save outputs to file
    with open(outdir + outfile + outext, "wb") as f:
        pickle.dump({"n_nodes" : n_nodes,
         "n_EA" : n_EA,
         "n_blocks" : n_blocks,
         "min_degree" : min_degree, 
         "starting_points" : starting_points,
         "graph_width" : graph_width,
         "p_in_left" : p_in_left, 
         "p_out_left" : p_out_left, 
         "p_in_right" : p_in_right, 
         "p_out_right" : p_out_right,
         "EA_lambda" : EA_lambda,
         "n_louvain" : n_louvain,
         "window_sizes" : window_sizes,
         "filepath" : filepath,
         "filebase" : filebase,
         "fileext" : fileext,
         "outdir" : outdir, 
         "outfile" : outfile,
         "outext" : outext,
         "split_time" : split_time,
         "AMI_oracle" : AMI_oracle,
         "AMI_unsubset" : AMI_unsubset,
         "AMI_unsubset_unrolled" : AMI_unsubset_unrolled,
         "AMI_CG" : AMI_CG,
         "AMI_windows" : AMI_windows, 
         "CG_range" : CG_range,
         "CG_range_total" : CG_range_total,
         "AMI_CG_total" : AMI_CG_total}, f)       
        
    
    
