"""
Jeremy D. Wendt
jdwendt@sandia.gov

    CircularList.py

    This code implements a circular list as described in an answer from 
    https://stackoverflow.com/questions/4151320/efficient-circular-buffer seen on 
    08/21/19.  This is in support of code for the paper Wendt, Field, Phillips, Wilson, 
    Soundarajan, and Bhowmick, "Partitioning Communication Streams into Graph Snapshots",
    2020.
"""


class CircularList(object):
    def __init__(self, size, data = []):
        """Initialization"""
        self.index = 0
        self.size = size
        self._data = list(data)[-size:]

    def append(self, value):
        """Append an element"""
        if len(self._data) == self.size:
            self._data[self.index] = value
        else:
            self._data.append(value)
        self.index = (self.index + 1) % self.size
        # print(self.index)

    def __getitem__(self, key):
        """Get element by index, relative to the current index"""
        if len(self._data) == self.size:
            return(self._data[(key + self.index) % self.size])
        else:
            return(self._data[key])

    def __repr__(self):
        """Return string representation"""
        return self._data.__repr__() + ' (' + str(len(self._data))+' items)'

    def __len__(self):
        return len(self._data)