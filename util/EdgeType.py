"""
Jeremy D. Wendt
jdwendt@sandia.gov

    EdgeType.py

    This code implements the Edge Type calculations as described in Section 3 of Wendt,
    Field, Phillips, Wilson, Soundarajan, and Bhowmick, "Partitioning Communication
    Streams into Graph Snapshots", 2020.
"""


from enum import IntEnum
import numpy


class EdgeType(IntEnum):
    """This enum allows the names from the paper be used in the code for each edge type"""
    N2 = 0
    N1 = 1
    N0 = 2
    R = 3


class EdgeData:
    """This class stores ancillary data about each edge as it is advertised"""
    def __init__(self, t, type):
        """Initialization"""
        self._first_time = self._last_time = t
        self._type = type
        self._num_times = 1

    def update(self, t):
        """A new advertisement is seen"""
        self._type = EdgeType.R
        self._last_time = t
        self._num_times += 1


class EdgeTyper:
    """This class stores the necessary information to compute edge types for each edge
    as it advertises."""
    def __init__(self):
        """Initialization"""
        self.__nodes = set()
        self.__edges = {}
        self.__edge_types = numpy.zeros(4)

    @staticmethod
    def canonical_order(src, dst):
        """Helper function that ensures edges are considered undirected"""
        if src < dst:
            return src, dst
        else:
            return dst, src

    def get_type(self, src, dst, time):
        """Returns the type for the input edge advertisement including updating internal
        state for repeat edges."""
        ordered = self.canonical_order(src, dst)
        # As this is by far the most common case in these datasets, make it the fastest
        # and first case.  (Considerable speed-up by having this first.)
        if ordered in self.__edges.keys():
            old_type = self.__edges[ordered]._type
            if old_type != EdgeType.R:
                self.__edge_types[int(old_type)] -= 1
                self.__edge_types[int(EdgeType.R)] += 1
            self.__edges[ordered].update(time)
            return EdgeType.R
        else:
            num_new = 0
            num_new += 1 if (src not in self.__nodes) else 0
            num_new += 1 if (dst not in self.__nodes) else 0
            self.__nodes.add(src)
            self.__nodes.add(dst)

            if num_new == 2:
                ret_type = EdgeType.N2
            elif num_new == 1:
                ret_type = EdgeType.N1
            else:
                ret_type = EdgeType.N0

            self.__edges[ordered] = EdgeData(time, ret_type)
            self.__edge_types[int(ret_type)] += 1
            return ret_type

    def num_unique_nodes(self):
        """Returns how many unique nodes have been in advertisements thus far"""
        return len(self.__nodes)

    def num_unique_edges(self):
        """Returns how many unique edges have been advertised thus far"""
        return len(self.__edges)

    def edges(self):
        """Returns all edges seen thus far"""
        return self.__edges

    def edge_proportions(self):
        """Computes the proportions seen for each edge type as a proportion of the graph
        size.  Not used in the paper."""
        inv_size = 1.0 / len(self.__edges)
        return [ x * inv_size for x in self.__edge_types ]

