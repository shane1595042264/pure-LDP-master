import math
import random

from pure_ldp.core import FreqOracleServer


class UOUEServer(FreqOracleServer):
    def __init__(self, epsilon, d, Xs, index_mapper=None):
        """
        Args:
            epsilon: float - the privacy budget
            d: integer - the size of the data domain
            use_oue: Optional boolean - If True, will use Optimised Unary Encoding (OUE)
            index_mapper: Optional function - maps data items to indexes in the range {0, 1, ..., d-1} where d is the size of the data domain
        """
        super().__init__(epsilon, d, index_mapper=index_mapper)
        self.set_name("UOUEServer")
        self.update_params(epsilon, d, index_mapper)
        self.Xs = Xs
    #Override
    def update_params(self, epsilon=None, d=None, index_mapper=None):
        """
        Updates UE server parameters. This will reset any aggregated/estimated data
        Args:
            epsilon: optional - privacy budget
            d: optional - domain size
            index_mapper: optional - index_mapper
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

    def aggregate(self, priv_data):
        """
        Used to aggregate privatised data by ue_client.privatise

        Args:
            priv_data: privatised data from ue_client.privatise
        """
        self.aggregated_data += priv_data
        self.n += 1

    def _update_estimates(self):
        self.estimated_data = (self.aggregated_data - self.n * self.d1) / (1 - 2 * self.d1) #(p - q), since q is d1, p = 1-q

        return self.estimated_data
#  No Xn because the data would stay the same anyway.

    def check_and_update_estimates(self):
        """
        Used to check if the "cached" estimated data needs re-estimating, this occurs when new data has been aggregated since last
        """
        if self.last_estimated < self.n:  # If new data has been aggregated since the last estimation, then estimate all
            self.last_estimated = self.n
            self._update_estimates()

    def estimate(self, data, suppress_warnings=False):
        """
        Calculates a frequency estimate of the given data item

        Args:
            data: data item
            suppress_warnings: Optional boolean - Supresses warnings about possible inaccurate estimations

        Returns: float - frequency estimate

        """
        self.check_warnings(suppress_warnings=suppress_warnings)
        index = self.index_mapper(data)

        if data not in self.Xs:
            return self.aggregated_data[index]/(1-self.d2) # Estimation from equation (10) in 4.2


        self.check_and_update_estimates()
        return self.estimated_data[index]
