"""
Jeremy D. Wendt
jdwendt@sandia.gov

    CreateGraphs.py

    This code implements Algorithm 1 described in Wendt, Field, Phillips, Prasadan, Wilson, 
    Soundarajan, and Bhowmick, "Partitioning Communication Streams into Graph Snapshots",
    2020.  This code contains many extra analyses for creating plots for the paper beyond
    those described in the simplified Algorithm 1 in the paper.

    Note that beyond the output files, considerable information is printed to stdout and
    should be piped to an output file.  That output has the following information:
    AdvertId,Timestamp,WindowSize,WindowedAdPropN2,WindowedAdPropN1,WindowedAdPropN0,
    WindowedAdPropR,NumNodes,NumEdges,WindExpectedMoreNodesAt100,
    SmoothedWindExpectedMoreNodesAtK,DerivWindExpectedMoreNodesAtK,
    WindExpectedMoreEdgesAtK,SmoothedWindExpectedMoreEdgesAtK,
    DerivWindExpectedMoreEdgesAtK,WindNodePredictionErrorAtLastK,
    SmoothedWindNodePredictionErrorAtLastK,WindEdgePredictionErrorAtLastK,
    SmoothedWindEdgePredictionErrorAtLastK,FirstCut
    where K in several of the above is replaced with predict_forward_ads's value.

# Usage: 
    python CreateGraphs.py "Input Filename" "Output Filename Base"

# Inputs:
    - Input filename (edge file) - A CSV of src,dst,timestamp values.  This must be sorted
    in increasing timestamp order, and this assumes there are no repeat events (same src,
    dst, and timestamp).  No header-line is expected or handled.
    - Output filename (Base) - Will be used to create output files: An overview file with
    the following information: Id,NumAds,NumNodes,NumEdges.  A static graph file for each
    snapshot with the following informaiton: node_a,node_b,first_time,last_time,num_times.

# Depends:
    - util/EdgeType
    - util/CircularList
"""


import sys

from util.EdgeType import EdgeTyper
from util.EdgeType import EdgeType
from util.CircularList import CircularList

import collections

# Run file directly from terminal
if __name__ == '__main__':
    
    # Arguments: "Input Filename" "Output Filename Base"
    if len(sys.argv) != 3:
        print('Expected arguments: input_file output_base')
        sys.exit(-1)
    
    # Parameters
    recent_history_max = 5000
    predict_forward_ads = 100
    derivative_window = 10000

    wind_edge_predictions = CircularList(predict_forward_ads)
    wind_node_predictions = CircularList(predict_forward_ads)
    wind_edge_pred_scores = CircularList(derivative_window)
    wind_node_pred_scores = CircularList(derivative_window)
    wind_edge_error_scores = CircularList(derivative_window)
    wind_node_error_scores = CircularList(derivative_window)
    wind_edge_error_num = wind_node_error_num = wind_edge_pred_num = wind_node_pred_num = 0
    node_derivative_scores = CircularList(derivative_window)
    edge_derivative_scores = CircularList(derivative_window)
    node_derivative_sum = edge_derivative_sum = 0

    # Get started with the actual graph data
    edge_typer = EdgeTyper()
    windowed_ad_history = [0 for x in range(4)]
    windowed_deque = collections.deque()

    num_ads = 0
    cut = 100
    first_cut = True
    # If not null, then files will be written out and new sufficient windows will be
    # started at each cut
    last_cut = 0
    file_number = 0

    print('AdvertId,Timestamp,WindowSize,WindowedAdPropN2,WindowedAdPropN1,WindowedAdPropN0,WindowedAdPropR,NumNodes,NumEdges,WindExpectedMoreNodesAt'
          + str(predict_forward_ads)+',SmoothedWindExpectedMoreNodesAt'
          + str(predict_forward_ads)+',DerivWindExpectedMoreNodesAt'
          + str(predict_forward_ads)+',WindExpectedMoreEdgesAt' 
          + str(predict_forward_ads)+',SmoothedWindExpectedMoreEdgesAt'
          + str(predict_forward_ads)+',DerivWindExpectedMoreEdgesAt'
          + str(predict_forward_ads)+',WindNodePredictionErrorAtLast'
          + str(predict_forward_ads)+',SmoothedWindNodePredictionErrorAtLast'
          + str(predict_forward_ads)+',WindEdgePredictionErrorAtLast'
          + str(predict_forward_ads)+',SmoothedWindEdgePredictionErrorAtLast'
          + str(predict_forward_ads)+',FirstCut')
    
    # File IO parameters
    input_file = open(sys.argv[1], 'r')
    write_out_files = sys.argv[2]
    for line in input_file:
        # Parse the data and check that it's good
        vals = line.strip().split(',')
        if len(vals) != 3:
            sys.stderr.write('Unexpected line does not have 3 elements: ' + line + '\n')
            sys.stderr.flush()
            exit(-2)

        # Handle windowed ad proportion ... first determine if it's time to prune the 
        # window
        # Classify edge type
        edge_type = edge_typer.get_type(vals[0], vals[1], vals[2])
        num_ads += 1 # Increment edge advertisement counter
        
        # Only consider latest recent_history_size advertisements
        # If current window is too large, toss out oldest advertisement
        recent_history_size = min(num_ads - last_cut, recent_history_max)
        if len(windowed_deque) == recent_history_size:
            windowed_ad_history[windowed_deque[0]] -= 1
            windowed_deque.popleft()
        # Store latest edge advertisement, update probabilities
        windowed_ad_history[int(edge_type)] += 1
        windowed_deque.append(edge_type)
        windowed_ad_percs = [ x / recent_history_size for x in windowed_ad_history ]

        # Now compute the predictions (and errors when applicable)
        mp = windowed_ad_percs[int(EdgeType.R)]
        nq = windowed_ad_percs[int(EdgeType.N1)]
        nr = windowed_ad_percs[int(EdgeType.N2)]
        
        # Expected number of new nodes and edges at next step
        wind_edge_mean = predict_forward_ads * (1 - mp)
        wind_node_mean = predict_forward_ads * (nq + 2 * nr)

        # Maintain the scores to smooth the predictions
        # Only consider the latest derivative_window events for smoothing
        if len(wind_edge_pred_scores) == derivative_window:
            wind_edge_pred_num -= wind_edge_pred_scores[0]
            wind_node_pred_num -= wind_node_pred_scores[0]
            node_derivative_sum -= node_derivative_scores[0]
            edge_derivative_sum -= edge_derivative_scores[0]

        # Update scores
        wind_edge_pred_num += wind_edge_mean
        wind_edge_pred_scores.append(wind_edge_mean)
        wind_node_pred_num += wind_node_mean
        wind_node_pred_scores.append(wind_node_mean)
        
        # Compute derivatives
        wind_len = len(wind_edge_pred_scores)
        deriv = -1 if wind_len == 1 else wind_node_pred_scores[wind_len - 1] - wind_node_pred_scores[wind_len - 2]
        node_derivative_sum += deriv
        node_derivative_scores.append(deriv)
        deriv = -1 if wind_len == 1 else wind_edge_pred_scores[wind_len - 1] - wind_edge_pred_scores[wind_len - 2]
        edge_derivative_sum += deriv
        edge_derivative_scores.append(deriv)

        # Maintain the scores for errors (and smoothed versions)
        wind_node_err = wind_edge_err = -1
        if len(wind_edge_predictions) == predict_forward_ads:
            wind_edge_err = wind_edge_predictions[0] - edge_typer.num_unique_edges()
            wind_node_err = wind_node_predictions[0] - edge_typer.num_unique_nodes()

            # Only consider the latest derivative_window events for smoothing
            if len(wind_edge_error_scores) == derivative_window:
                wind_edge_error_num -= wind_edge_error_scores[0]
                wind_node_error_num -= wind_node_error_scores[0]

            wind_edge_error_num += wind_edge_err
            wind_edge_error_scores.append(wind_edge_err)
            wind_node_error_num += wind_node_err
            wind_node_error_scores.append(wind_node_err)

        wind_edge_predictions.append(wind_edge_mean + edge_typer.num_unique_edges())
        wind_node_predictions.append(wind_node_mean + edge_typer.num_unique_nodes())

        # Make sure the denominator isn't 0
        smooth_denom_pred = max(1, len(wind_edge_pred_scores))
        smooth_denom_err = max(1, len(wind_edge_error_scores))
        smooth_node_deriv = node_derivative_sum / smooth_denom_pred
        smooth_edge_deriv = edge_derivative_sum / smooth_denom_pred

        # Determine whether or not to cut (actual cut happens after print outs)
        # NOTE: Precision issues for some tests required using not strictly 0 here
        if smooth_node_deriv >= -1e-14 and smooth_edge_deriv >= -1e-14 and first_cut:
            cut *= -1
            first_cut = False
            
        print('%d,%s,%d,%lf,%lf,%lf,%lf,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%d' %
              ((num_ads-1),vals[2],recent_history_size,windowed_ad_percs[int(EdgeType.N2)],windowed_ad_percs[int(EdgeType.N1)],
               windowed_ad_percs[int(EdgeType.N0)],windowed_ad_percs[int(EdgeType.R)],edge_typer.num_unique_nodes(),
               edge_typer.num_unique_edges(),wind_node_mean,wind_node_pred_num/smooth_denom_pred,smooth_node_deriv,wind_edge_mean,
               wind_edge_pred_num/smooth_denom_pred,smooth_edge_deriv,wind_node_err,wind_node_error_num/smooth_denom_err,wind_edge_err,
               wind_edge_error_num/smooth_denom_err, cut))
        
        if num_ads % 100000 == 0:
            sys.stderr.write('Completed %d timestamps\n' % (num_ads))
            sys.stderr.flush()

        # Now, handle the case where I'm storing out all cuts and reset state
        if not first_cut and write_out_files is not None:
            with open(write_out_files + '_overview.csv', 'a') as overview_file:
                if file_number == 0:
                    overview_file.write('Id,NumAds,NumNodes,NumEdges\n')
                overview_file.write('%d,%d,%d,%d\n' % (file_number, num_ads-last_cut,edge_typer.num_unique_nodes(),edge_typer.num_unique_edges()))
            
            # Write current graph to file
            with open(write_out_files + '_graph_' + str(file_number) + '.csv', 'w') as edge_file:
                edge_file.write('node_a,node_b,first_time,last_time,num_times\n')
                for edge, metadata in edge_typer.edges().items():
                    edge_file.write('%s,%s,%s,%s,%d\n' % (edge[0], edge[1], metadata._first_time, metadata._last_time, metadata._num_times))
           
            # Now reset state: end one graph start the next
            wind_edge_predictions = CircularList(predict_forward_ads)
            wind_node_predictions = CircularList(predict_forward_ads)
            wind_edge_pred_scores = CircularList(derivative_window)
            wind_node_pred_scores = CircularList(derivative_window)
            wind_edge_error_scores = CircularList(derivative_window)
            wind_node_error_scores = CircularList(derivative_window)
            wind_edge_error_num = wind_node_error_num = wind_edge_pred_num = wind_node_pred_num = 0
            node_derivative_scores = CircularList(derivative_window)
            edge_derivative_scores = CircularList(derivative_window)
            node_derivative_sum = edge_derivative_sum = 0
            edge_typer = EdgeTyper()
            windowed_ad_history = [ 0 for x in range(4) ]
            windowed_deque = collections.deque()
            first_cut = True
            last_cut = num_ads
            file_number += 1

    input_file.close()


