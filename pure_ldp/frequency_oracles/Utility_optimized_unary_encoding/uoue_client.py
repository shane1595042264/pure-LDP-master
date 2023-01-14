import numpy as np
import math
import random

from pure_ldp.core import FreqOracleClient


# Client-side for unary-encoding
    # By default parameters are set for Symmetric Unary Encoding (SUE)
    # If is_oue=True is passed to the constructor then it uses Optimised Unary Encoding (OUE)

class UOUEClient(FreqOracleClient):
    #FreqOracleClient is in _freq_oracle_client.py in pure_ldp\core
    #d: how many bit vectors.
    def __init__(self, epsilon, d, Xs, index_mapper=None):
        """

        Args:
            epsilon: float - privacy budget
            d: integer - the size of the data domain
            use_oue: Optional boolean - if True, will use Optimised Unary Encoding (OUE)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.update_params(epsilon, d, index_mapper)
        self.Xs = Xs

    def update_params(self, epsilon=None, d=None, index_mapper=None):
        """
        Used to update the client UOUE parameters.
        Args:
            epsilon: optional - privacy budget
            d: optional - domain size
            index_mapper:  optional - function
        """
        super().update_params(epsilon, d, index_mapper)

        if epsilon is not None:
            const1 = math.pow(math.e, self.epsilon / 2)
            # const = e^(epsilon/2)
            self.theta = const1 / (const1 + 1)

            self.d1 = self.d1Construct(epsilon, self.theta)
            self.d2 = self.d2Construct(epsilon, self.theta)

    def d1Construct(self, epsilon, theta):
        const = math.pow(math.e, epsilon)
        d1 = theta / ((1 - theta) * const + theta)

        return d1

    def d2Construct(self, epsilon, theta):
        const = math.pow(math.e, epsilon)
        d2 = ((1-theta)*const+theta) / const
        return d2



    def _perturbXs(self, index):
        """
        Used internally to peturb data using unary encoding

        Args:
            index: the index corresponding to the data item

        Returns: privatised data vector

        """
        oh_vec = np.random.choice([1, 0], size=self.d, p=[self.d1, 1 - self.d1])  # If entry is 0, flip with prob q
        if random.random() < self.theta:
            oh_vec[index] = 1
        else:
            oh_vec[index] = 0

        return oh_vec
    # it deals with each input separately.
    def _perturbXn(self, index):
        """
        Used internally to peturb data using unary encoding

        Args:
            index: the index corresponding to the data item

        Returns: privatised data vector

        """

        oh_vec = np.random.choice([1, 0], size=self.d, p=[0, 1])  # If entry is 0, flip with prob q

        # d may just be the number of rows in the data.
        if random.random() < self.d2:
            oh_vec[index] = 0
        else:
            oh_vec[index] = 1
        return oh_vec

    def privatise(self, data):
        """
        Privatises a user's data item using unary encoding.

        Args:
            data: data item

        Returns: privatised data vector

        """

        index = self.index_mapper(data)

        if data in self.Xs:
            return self._perturbXs(index)
        else:
            return self._perturbXn(index)

