"""

Richard V. Field, Jr.
rvfield@sandia.gov

    SyntheticData.py

    This code implements the sythetic data generator described in Appendix B of Wendt,
    Field, Phillips, Prasadan, Wilson, Soundarajan, and Bhowmick, "Partitioning 
    Communication Streams into Graph Snapshots", 2020.  The output file
    (syntheticData_adList_overlap_OVERLAP_transitionRatePercent_TRANSITION_RATE_PERCENT.csv)
    will match the expected format for CreateGraphs.py (src,dst,timestamp)

# Usage:
    python SyntheticData.py overlap transitionRatePercent

# Inputs:
    - overlap - A floating point number for percent of overlap, \in (0, 100)
    - transitionRatePercent - A floating point number for percent of total sim time it
            takes to completely transition, \in (1, 100)

# Details:
    Consider a stochastic block model (SBM) with 10 blocks and n total nodes, and
    let A denote the n x n adjacency matrix of the SBM. We define subgraphs G0 and
    G1 as follows.

    Let r \in {0, 1, ..., n/4} be an integer. G0 and G1 are defined by the n/2 x n/2
    top left and bottom right submatrices of A, offset by r, that is:

        A0 = A[    r: r+n/2,     r: r+n/2],
        A1 = A[n/2-r:   n-r, n/2-r:   n-r]

    are the adjacency matrices of G0 and G1, respectively. The proportion of the
    graphs that overlap is:

                       2        2
                 ( 2r )     16 r
      overlap = -------  = -----
                       2      2
                ( n/2 )      n

    If the overlap \in (0, 1) is specified, then:

           n
      r = --- SQRT(overlap)
           4

    Let E0 and E1 denote the edge sets from graphs G0 and G1, where we note that their
    intersection is not empty in general. Let 0 < t0 < t1 denote two transition times.
    We form a dynamic undirected graph from E0 and E1 as follows:

    For all times t, we build edge advertisements from edge set E0 with probability
    p0(t) and from edge set E1 with probability 1 - p0(t), where

            { 1,          t <= t0
            {
            { (t - t1)
    p0(t) = { --------,   t0 < t <= t1
            { (t0 - t1)
            {
            { 0,          t > t1

    All edge advertisements follow a homogeneous Poisson process with intensity lam.

    This implies:

    (1) For all times t <= t0, we build edges from G0 with probability one.
    (2) For all times t >= t1, we build edges from G1 with probability one.
    (3) For all times t such that t0 < t < t1, we we build edges from graph G0
        with probability p0(t) and from graph G1 with probability 1 - p0(t).
    (4) Edge advertisement occur at an average rate of lam per unit time.

"""

import numpy as np
import networkx as nx
import csv
import sys

# Run file directly from terminal
if __name__ == '__main__':
    overlap               = float(sys.argv[1])
    transitionRatePercent = float(sys.argv[2])
    print('-------------------------------------')
    print('Running script ' + sys.argv[0] + ' with inputs:')
    print('              overlap = ' + str(overlap))
    print('transitionRatePercent = ' + str(transitionRatePercent))
    print('')

    #---Parameters
    #-- note: last ad time will be approximately numAds/lam
    numberOfBlocks = 10
    nodesPerBlock  = 200
    lam            = 25       # mean rate of edge advertisments
    t0             = 1000     # start time of 'transition region'
    numAds         = 100000   # total number of edge advertisements
    P              = 0.05     # SBM on-diagonal probabilities
    q              = 0.01     # SBM off-diagonal probabilities
    nseed          = 123      # seed to RNG for SBM

    #---Build the SBM
    sizes = [nodesPerBlock for b in range(numberOfBlocks)]

    # the "planted partition assumption" - all on-/off-diagonal entries are p/q
    probs = [[ P, q, q, q, q, q, q, q, q, q], \
             [ q, P, q, q, q, q, q, q, q, q], \
             [ q, q, P, q, q, q, q, q, q, q], \
             [ q, q, q, P, q, q, q, q, q, q], \
             [ q, q, q, q, P, q, q, q, q, q], \
             [ q, q, q, q, q, P, q, q, q, q], \
             [ q, q, q, q, q, q, P, q, q, q], \
             [ q, q, q, q, q, q, q, P, q, q], \
             [ q, q, q, q, q, q, q, q, P, q], \
             [ q, q, q, q, q, q, q, q, q, P]]
    G = nx.stochastic_block_model(sizes, probs, seed=nseed)

    # Pull out edge sets E0 and E1
    numberOfNodes  = numberOfBlocks * nodesPerBlock
    r              = int(round(numberOfNodes / 4 * np.sqrt(overlap/100)))

    N0 = [u for u in range(r, r + int(numberOfNodes/2))]
    N1 = [u for u in range(int(numberOfNodes/2) - r, numberOfNodes - r)]
    n0 = len(N0)
    n1 = len(N1)
    E0 = list(G.subgraph(N0).edges)
    E1 = list(G.subgraph(N1).edges)
    m0 = len(E0)
    m1 = len(E1)

    numSharedNodes = len(set(N0).intersection(set(N1)))
    numSharedEdges = len(set(E0).intersection(set(E1)))

    print('G is a SBM with 10 blocks and: (p, q) = (' + str(P) + ', ' + str(q) + ').')
    print('Number of nodes, edges in G0: ' + str(n0) + ', ' + str(m0))
    print('Number of nodes, edges in G1: ' + str(n1) + ', ' + str(m1))
    print('Number of shared nodes: ' + str(numSharedNodes))
    print('Number of shared edges: ' + str(numSharedEdges))

    # end time of 'transition region'
    t1 = t0 + (transitionRatePercent * numAds / (100 * lam))

    print('Transition zone is: [' + str(t0) + ', ' + str(t1) + '].')

    # Simulate edge advertisement times
    np.random.seed(seed=nseed)
    #-- interarrivals
    edgeAdIAT = np.random.exponential(1/lam, size=numAds)
    #-- arrivals
    edgeAdArrivalTimes = np.cumsum(edgeAdIAT)

    print('Last ad time is: ' + str(edgeAdArrivalTimes[-1]))
    print('-------------------------------------')
    print('')

    # Create edges
    adList  = []
    firstAd = True
    lastAd  = True
    for ad in range(0, numAds):

        # ad time
        t = edgeAdArrivalTimes[ad]

        if t <= t0:
            # draw from E0
            idx  = np.random.randint(m0)
            u, v = E0[idx]
        elif t > t1:
            if lastAd:
                print('Leaving transition zone at ad: ' + str(ad) + ' and time ' + str(t))
                lastAd = False
            # draw from E1
            idx  = np.random.randint(m1)
            u, v = E1[idx]
        else:
            if firstAd:
                print('Entering transition zone at ad: ' + str(ad) + ' and time ' + str(t))
                firstAd = False
            # draw from E0 / E1 with probability p0 / 1-p0
            p0 = (t - t1) / (t0 - t1)
            r  = np.random.uniform()
            if r <= p0:
                idx  = np.random.randint(m0)
                u, v = E0[idx]
            else:
                idx  = np.random.randint(m1)
                u, v = E1[idx]

        # save the advertisement
        adList.append((u, v, t))

    # save ad list to CSV
    fName = 'syntheticData_adList_overlap_' + str(int(overlap)) + \
             '_transitionRatePercent_' + str(int(transitionRatePercent)) + '.csv'
    with open(fName, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(adList)
