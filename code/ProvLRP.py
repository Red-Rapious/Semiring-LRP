import numpy as np
import numbers

class MultipleInitialisation(Exception):
    pass

class Semiring():
    def handle_initialisation(self, not_none, n1, n2, n3):
        if not_none is not None:
                    assert isinstance(not_none, numbers.Number), f"must be a number, is of type {type(not_none)}"
                    self.value = not_none
                    if n1 is not None or n2 is not None or n3 is not None:
                        raise MultipleInitialisation
    
    def __init__(self, relevance=None, activation=None, weight=None, standard=None):
        self.handle_initialisation(relevance, activation, weight, standard)
        self.handle_initialisation(activation, relevance, weight, standard)
        self.handle_initialisation(weight, activation, relevance, standard)
        self.handle_initialisation(standard, activation, weight, relevance)

    def __mul__(self, b):
        f = self.value * b.value
        s = Semiring()
        s.value = f
        return s
    
    def to_float(self):
        return self.value
    
    def __add__(self, b):
        f = self.value + b.value
        s = Semiring()
        s.value = f
        return s
    
    def __sub__(self, b):
        f = self.value - b.value
        s = Semiring()
        s.value = f
        return s
    
    def pos(self):
        f = max(0, self.value)
        s = Semiring()
        s.value = f
        return s

    def neg(self):
        f = min(0, self.value)
        s = Semiring()
        s.value = f
        return s
    
    def lower_bound(A, bound):
        return np.vectorize(identity)(bound*np.ones(A[0].shape))

    def upper_bound(A, bound):
        return np.vectorize(identity)(bound*np.ones(A[0].shape))
    
#from_relevance = np.vectorize(lambda x : Semiring(relevance=x))
from_activation = np.vectorize(lambda x : Semiring(activation=x))
from_weight = np.vectorize(lambda x : Semiring(weight=x))
identity = np.vectorize(lambda x : Semiring(standard=x))
pos = np.vectorize(lambda x : x.pos())
neg = np.vectorize(lambda x : x.neg())

def from_relevance(m):
    return np.array([[Semiring(activation=m[x][y]) for y in range(m.shape[1])] for x in range(m.shape[0])])

def semiring_to_float(m):
    return np.array([[m[x][y].to_float() for y in range(m.shape[1])] for x in range(m.shape[0])])