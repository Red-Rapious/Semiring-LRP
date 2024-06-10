import numpy as np
import numbers
import ProvLRP_config

class MultipleInitialisation(Exception):
    pass

class Semiring():
    def handle_initialisation(self, not_none, n1, n2):
        if not_none is not None:
                    assert isinstance(not_none, numbers.Number), f"must be a number, is of type {type(not_none)}"
                    self.value = not_none
                    if n1 is not None or n2 is not None:
                        raise MultipleInitialisation
                    
    def handle_relevance(self, relevance, n1, n2):
         # for other semiring, apply a transform to the first argument
         self.handle_initialisation(relevance, n1, n2)

    def handle_activation(self, activation, n1, n2):
         # for other semiring, apply a transform to the first argument
         self.handle_initialisation(activation, n1, n2)
    
    def handle_weight(self, weight, n1, n2):
         # for other semiring, apply a transform to the first argument
         self.handle_initialisation(weight, n1, n2)

    def __init__(self, relevance=None, activation=None, weight=None):
        self.handle_relevance(relevance, activation, weight)
        self.handle_activation(activation, relevance, weight)
        self.handle_weight(weight, activation, relevance)

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
    
    def lower_bound(shape, bound):
        return np.vectorize(Semiring.from_activation)(bound*np.ones(shape))

    def upper_bound(shape, bound):
        return np.vectorize(Semiring.from_activation)(bound*np.ones(shape))
    
    from_activation = np.vectorize(lambda x : Semiring(activation=x))
    from_weight = np.vectorize(lambda x : Semiring(weight=x))
    vect_pos = np.vectorize(lambda x : x.pos())
    vect_neg = np.vectorize(lambda x : x.neg())

    def from_relevance(m):
        return np.array([[Semiring(activation=m[x][y]) for y in range(m.shape[1])] for x in range(m.shape[0])])

    def semiring_to_float(m):
        return np.array([[m[x][y].to_float() for y in range(m.shape[1])] for x in range(m.shape[0])])
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    

class BooleanSemiring(Semiring):
    def threshold(value, theta):
        if value is None:
            return None
        return value >= theta

    def handle_initialisation(self, not_none, n1, n2):
        if not_none is not None:
            # TODO: test somewhere else in the pipeline that
            # the parameter is correct (below, the parameter)
            # is already transformed in semiring element, so it's too late
            #assert isinstance(not_none, numbers.Number), f"must be a number, is of type {type(not_none)}"
            self.value = not_none
            if n1 is not None or n2 is not None:
                raise MultipleInitialisation
                    
    def handle_relevance(self, relevance, n1, n2):
        relevance_b = BooleanSemiring.threshold(relevance, ProvLRP_config.boolean_relevance_threshold)
        self.handle_initialisation(relevance_b, n1, n2)

    def handle_activation(self, activation, n1, n2):
        activation_b = BooleanSemiring.threshold(activation, ProvLRP_config.boolean_activation_threshold)
        self.handle_initialisation(activation_b, n1, n2)
    
    def handle_weight(self, weight, n1, n2):
        weight_b = BooleanSemiring.threshold(weight, ProvLRP_config.boolean_weight_threshold)
        self.handle_initialisation(weight_b, n1, n2)

    def __init__(self, relevance=None, activation=None, weight=None):
        self.handle_relevance(relevance, activation, weight)
        self.handle_activation(activation, relevance, weight)
        self.handle_weight(weight, activation, relevance)

    def __mul__(self, b):
        f = self.value and b.value
        s = BooleanSemiring()
        s.value = f
        return s
    
    def to_float(self):
        return 1 if self.value else 0
    
    def __add__(self, b):
        f = self.value or b.value
        s = BooleanSemiring()
        s.value = f
        return s
    
    def __sub__(self, b):
        print("Warning, should not be used without zB rule")
        f = self.value and not b.value
        s = BooleanSemiring()
        s.value = f
        return s
    
    from_activation = np.vectorize(lambda x : BooleanSemiring(activation=x))
    from_weight = np.vectorize(lambda x : BooleanSemiring(weight=x))

    def from_relevance(m):
        return np.array([[BooleanSemiring(activation=m[x][y]) for y in range(m.shape[1])] for x in range(m.shape[0])])


class TropicalSemiring(Semiring):
    def handle_initialisation(self, not_none, n1, n2):
        if not_none is not None:
            assert isinstance(not_none, numbers.Number), f"must be a number, is of type {type(not_none)}"
            self.value = max(0, not_none)
            if n1 is not None or n2 is not None:
                raise MultipleInitialisation
                    
    def handle_relevance(self, relevance, n1, n2):
        self.handle_initialisation(relevance, n1, n2)

    def handle_activation(self, activation, n1, n2):
        self.handle_initialisation(activation, n1, n2)
    
    def handle_weight(self, weight, n1, n2):
        self.handle_initialisation(weight, n1, n2)

    def __init__(self, relevance=None, activation=None, weight=None):
        self.handle_relevance(relevance, activation, weight)
        self.handle_activation(activation, relevance, weight)
        self.handle_weight(weight, activation, relevance)

    def __mul__(self, b):
        f = self.value + b.value
        s = TropicalSemiring()
        s.value = f
        return s
    
    def to_float(self):
        return self.value
    
    def __add__(self, b):
        f = min(self.value, b.value)
        s = TropicalSemiring()
        s.value = f
        return s
    
    from_activation = np.vectorize(lambda x : TropicalSemiring(activation=x))
    from_weight = np.vectorize(lambda x : TropicalSemiring(weight=x))

    def from_relevance(m):
        return np.array([[TropicalSemiring(activation=m[x][y]) for y in range(m.shape[1])] for x in range(m.shape[0])])
    

class CountingSemiring(Semiring):
    def threshold(value, theta):
        if value is None:
            return None
        return 1 if value >= theta else 0
    
    def handle_initialisation(self, not_none, n1, n2):
        if not_none is not None:
            assert isinstance(not_none, numbers.Number), f"must be a number, is of type {type(not_none)}"
            self.value = not_none
            if n1 is not None or n2 is not None:
                raise MultipleInitialisation
                    
    def handle_relevance(self, relevance, n1, n2):
        relevance_i = CountingSemiring.threshold(relevance, ProvLRP_config.counting_relevance_threshold)
        #relevance_i = max(0, int(relevance))
        self.handle_initialisation(relevance_i, n1, n2)

    def handle_activation(self, activation, n1, n2):
        activation_i = CountingSemiring.threshold(activation, ProvLRP_config.counting_activation_threshold)
        #activation_i = max(0, int(activation))
        self.handle_initialisation(activation_i, n1, n2)
    
    def handle_weight(self, weight, n1, n2):
        weight_i = CountingSemiring.threshold(weight, ProvLRP_config.counting_weight_threshold)
        #weight_i = max(0, int(weight))
        self.handle_initialisation(weight_i, n1, n2)

    def __init__(self, relevance=None, activation=None, weight=None):
        self.handle_relevance(relevance, activation, weight)
        self.handle_activation(activation, relevance, weight)
        self.handle_weight(weight, activation, relevance)

    def __mul__(self, b):
        f = self.value * b.value
        s = CountingSemiring()
        s.value = f
        return s
    
    def to_float(self):
        return self.value
    
    def __add__(self, b):
        f = self.value + b.value
        s = CountingSemiring()
        s.value = f
        return s
    
    def __sub__(self, b):
        print("Warning, should not be used without zB rule")
        f = max(0, self.value - b.value)
        s = CountingSemiring()
        s.value = f
        return s
    
    from_activation = np.vectorize(lambda x : CountingSemiring(activation=x))
    from_weight = np.vectorize(lambda x : CountingSemiring(weight=x))

    def from_relevance(m):
        return np.array([[CountingSemiring(activation=m[x][y]) for y in range(m.shape[1])] for x in range(m.shape[0])])