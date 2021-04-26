"""
We want to test community structure capabilities of SBMs sampled from EAs.  We need the
SBMs such that each node is guaranteed a minimum degree -- so that nodes are unlikely to
be missed by any reasonable window size.  (Otherwise AMI can be a problem when comparing
to systems with missing nodes.)  We do this by merging a SBM with a random graph where all
nodes have minimum degree.

@author: Rich V. Field, Jeremy D. Wendt, Arvind Prasadan

"""

import typing
import networkx as nx
import random
import numpy as np


def min_degree_sbm(num_nodes: int, num_blocks: int, p_in: float, p_out: float, 
                   min_degree: int, seed: int=0) -> nx.Graph:
    """
    This function creates a minimum degree SBM approximating the input parameters.  This
    requries that num_nodes / num_blocks is an integer division result.
    
    Note that when min_degree was 2, this failed on the assert for actual_degrees.  I
    don't know why, so we may need to remove that assert.
    """
    # First make the random graph with minimum-degree:
    degree_list = [min_degree for i in range(num_nodes)]
    rnd_G = nx.configuration_model(degree_list, seed=seed)
    
    # The returned graph is a multigraph, which may have parallel edges.  To remove any
    # parallel edges from the returned graph...
    rnd_G = nx.Graph(rnd_G)
    
    # check
    actual_degrees = [d for v, d in rnd_G.degree()]
    for i in range(len(actual_degrees)):
        if actual_degrees[i] != degree_list[i]:
            print('At %d, %d != %d' % (i, actual_degrees[i], degree_list[i]))
    assert actual_degrees == degree_list, 'Random graph generator failed'
    
    # Make the SBM
    num_nodes_per_block = int(num_nodes / num_blocks)
    assert num_nodes_per_block * num_blocks == num_nodes, 'Input number of nodes not '\
                                                         'divisible by number of blocks'
    
    sizes = [num_nodes_per_block for b in range(num_blocks)]
    probs = p_out * np.ones((num_blocks, num_blocks))
    for i in range(num_blocks):
        probs[i][i] = p_in
    
    SBM_G = nx.stochastic_block_model(sizes, probs, seed=seed)
    
    # Now merge them and return
    return merge_graphs(SBM_G, rnd_G)


def merge_graphs(g1: nx.Graph, g2: nx.Graph) -> nx.Graph:
    """
    A simple helper function that merges two graphs together.  Only used by min_degree_sbm
    for now, but could change over time.  By "merge", we mean that all edges in both
    input graphs are combined into a returned graph
    """
    edge_list_merged = list(set(g1.edges).union(set(g2.edges)))
    return nx.Graph(edge_list_merged)


def reorder_nodes(graph: nx.Graph, seed: int=0) -> (nx.Graph, typing.Dict):
    """
    This helper re-names all the nodes in the input graph to a random other set of names.
    This is necessary for our test so that we can use the same node set but have two 
    distinct and disjoint community detection results.
    """
    # First, create the random reordering for the nodes
    num_nodes = graph.number_of_nodes()
    ordered = [i for i in range(num_nodes)]
    random.shuffle(ordered)
    
    # Now, rename all the nodes and edges
    mapping = {}
    for i, val in enumerate(ordered):
        mapping[i] = val
    return nx.relabel_nodes(graph, mapping, copy=True), mapping


def partition_ids_sbm(num_nodes: int, num_blocks: int, mapping: typing.Dict=None) -> typing.List:
    """
    This helper creates the correct answer partitions for an SBM with the input 
    parameters.  If mapping isn't None, this correctly fixes the SBM assignments for the
    mapping.  It's assumed that mapping came from reorder_nodes.
    """
    ids = [int((i / num_nodes) * num_blocks) for i in range(num_nodes)]
    if mapping is not None:
        tmp = [ids[i] for i in range(num_nodes)]
        for key, value in mapping.items():
            tmp[value] = ids[key]
        ids = tmp
    
    return ids