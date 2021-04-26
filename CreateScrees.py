"""
Arvind Prasadan
aprasad@sandia.gov
Jeremy D. Wendt
jdwendt@sandia.gov

    CreateScrees.py

    This code creates scree plots described in Wendt, Field, Phillips, Prasadan, Wilson, 
    Soundarajan, and Bhowmick, "Partitioning Communication Streams into Graph Snapshots",
    2020.

# Usage: 
    python CreateScrees.py "Input Filename" "Output Filename"

# Inputs:
    - Input filename - The name for the snapshot similarity files output by 
    MergeGraphs.py.  If inputs do not have the expected header-line (id_x,id_y,
    nodes_jaccard,edges_jaccard,nodes_cosine,edges_cosine,nodes_wjaccard,edges_wjaccard), 
    this will fail.
    - Output filename - The name for output file with the number of merged snapshots at
    each threshold possible for each similarity measure as follows:
    NodeCosThresh,NodeCosScree,EdgeCosThresh,EdgeCosScree,NodeJaccThresh,NodeJaccScree,EdgeJaccThresh,EdgeJaccScree,NodeWJaccThresh,NodeWJaccScree,EdgeWJaccThresh,EdgeWJaccScree

# Assumptions:
    - Input Files are created with MergeGraphs.py (formatting)
"""
import sys
import typing
import numpy as np
import math

def find_merge_groups(sim_mat:np.array, thresh: typing.List) -> typing.List:
    """
        find_merge_groups(sim_mat, thresh)

    # Inputs:
        - NumPy 2d array with all similarity values for the specified similarity measure
        - Python list with all similarities to be considered

    # Outputs:
        - Python list of tuples with (1) the threshold applied, and (2) the number of
        merged snapshots resulting from that threshold.

    # Depends:
        - math
        - numpy
    """
    assert 2 == len(sim_mat.shape), "Wrong number of dimensions"
    assert sim_mat.shape[0] == sim_mat.shape[1], "Matrix is not square"
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i][j] != sim_mat[j][i]:
                print('Not equal at %d, %d: %lf != %lf' % (i, j, sim_mat[i][j], sim_mat[j][i]))
    assert np.allclose(sim_mat, sim_mat.T), "Matrix is not symmetric"

    # Sort the thresholds:
    sorted_thresh = sorted(thresh)

    ret = []
    ret.append((0, 1))
    for thresh_val in sorted_thresh:
        # Apply threshold
        tmp_mat = (sim_mat > thresh_val)

        group_list = []
        group_start_idx = 0
        for idx in range(1, tmp_mat.shape[0]):
            if 1.0 > np.mean(tmp_mat[np.ix_(range(group_start_idx, idx + 1),
                                            range(group_start_idx, idx + 1))]):
                group_list += [list(range(group_start_idx, idx))]
                group_start_idx = idx
        group_list += [list(range(group_start_idx, idx + 1))]
        ret.append((thresh_val, min(sim_mat.shape[0], len(group_list))))

    return ret


# Run file directly from terminal
if __name__ == '__main__':
    # Arguments:
    # input_filename is an output similarities file from MergeGraphs.py
    # output_filename is the name for the csv this will output
    if len(sys.argv) != 3:
        print('Expected arguments: input_filename output_filename')
        sys.exit(-1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    # Read in arguments
    col_ids = []
    row_ids = []
    node_cos = []
    node_jacc = []
    node_wjacc = []
    edge_cos = []
    edge_jacc = []
    edge_wjacc = []
    first_line = True
    for line in open(input_filename, 'r'):
        if first_line:
            first_line = False
            if line.strip() != 'id_x,id_y,nodes_jaccard,edges_jaccard,nodes_cosine,edges_cosine,nodes_wjaccard,edges_wjaccard':
                print('Input file didn\'t match merge-file output:\n' + line)
                sys.exit(-1)
            continue
        vals = line.split(',')
        if len(vals) != 8:
            continue
        row_ids.append(int(vals[0]))
        col_ids.append(int(vals[1]))
        node_jacc.append(0 if math.isnan(float(vals[2])) else float(vals[2]))
        edge_jacc.append(0 if math.isnan(float(vals[3])) else float(vals[3]))
        node_cos.append(0 if math.isnan(float(vals[4])) else float(vals[4]))
        edge_cos.append(0 if math.isnan(float(vals[5])) else float(vals[5]))
        node_wjacc.append(0 if math.isnan(float(vals[6])) else float(vals[6]))
        edge_wjacc.append(0 if math.isnan(float(vals[7])) else float(vals[7]))

    # Convert arguments to 2d matrices
    max_idx = max(col_ids) + 1

    node_cos_mat = np.zeros((max_idx, max_idx))
    node_jacc_mat = np.zeros((max_idx, max_idx))
    node_wjacc_mat = np.zeros((max_idx, max_idx))
    edge_cos_mat = np.zeros((max_idx, max_idx))
    edge_jacc_mat = np.zeros((max_idx, max_idx))
    edge_wjacc_mat = np.zeros((max_idx, max_idx))

    for i in range(len(col_ids)):
        node_cos_mat[row_ids[i], col_ids[i]] = node_cos[i]
        node_cos_mat[col_ids[i], row_ids[i]] = node_cos[i]
        edge_cos_mat[row_ids[i], col_ids[i]] = edge_cos[i]
        edge_cos_mat[col_ids[i], row_ids[i]] = edge_cos[i]
        node_jacc_mat[row_ids[i], col_ids[i]] = node_jacc[i]
        node_jacc_mat[col_ids[i], row_ids[i]] = node_jacc[i]
        edge_jacc_mat[row_ids[i], col_ids[i]] = edge_jacc[i]
        edge_jacc_mat[col_ids[i], row_ids[i]] = edge_jacc[i]
        node_wjacc_mat[row_ids[i], col_ids[i]] = node_wjacc[i]
        node_wjacc_mat[col_ids[i], row_ids[i]] = node_wjacc[i]
        edge_wjacc_mat[row_ids[i], col_ids[i]] = edge_wjacc[i]
        edge_wjacc_mat[col_ids[i], row_ids[i]] = edge_wjacc[i]

    # Compute the values for the scree plots
    node_cos_scree = find_merge_groups(node_cos_mat, node_cos)
    edge_cos_scree = find_merge_groups(edge_cos_mat, edge_cos)
    node_jacc_scree = find_merge_groups(node_jacc_mat, node_jacc)
    edge_jacc_scree = find_merge_groups(edge_jacc_mat, edge_jacc)
    node_wjacc_scree = find_merge_groups(node_wjacc_mat, node_wjacc)
    edge_wjacc_scree = find_merge_groups(edge_wjacc_mat, edge_wjacc)

    # Write outputs to file
    outfile = open(output_filename, 'w')
    outfile.write('NodeCosThresh,NodeCosScree,EdgeCosThresh,EdgeCosScree,NodeJaccThresh,NodeJaccScree,EdgeJaccThresh,EdgeJaccScree,NodeWJaccThresh,NodeWJaccScree,EdgeWJaccThresh,EdgeWJaccScree\n')
    for i in range(len(node_cos_scree)):
        outfile.write('%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n' %
                      (node_cos_scree[i][0], node_cos_scree[i][1], edge_cos_scree[i][0], edge_cos_scree[i][1],
                       node_jacc_scree[i][0], node_jacc_scree[i][1], edge_jacc_scree[i][0], edge_jacc_scree[i][1],
                       node_wjacc_scree[i][0], node_wjacc_scree[i][1], edge_wjacc_scree[i][0], edge_wjacc_scree[i][1]))
    outfile.close()
