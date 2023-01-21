from pure_ldp.frequency_oracles import *
import numpy as np
from collections import Counter
from pure_ldp.frequency_oracles.Utility_optimized_unary_encoding import UOUEClient, UOUEServer

dataXn = np.concatenate(([1] * 8000, [2] * 4000, [3] * 1000, [4] * 500))
XnUnique = np.unique(dataXn)
dXn = XnUnique.size

# Xn_rows, Xn_cols = dataXn.shape
# print(dataXs)
dataXs = np.concatenate(([5] * 1000, [6] * 1800, [7] * 2000, [8] * 300))
XsUnique = np.unique(dataXs)
dXs = XsUnique.size

# ---------------------------------------Test---------------------------------------------------------

dataXs_Type = type(dataXs)
print("The type of DataXs is: ", dataXs_Type)

# ------------------------------------------------------------------------------------------------
Original_Xs_freq = list(Counter(dataXs).values())  # True frequencies of the dataset Xs
Original_Xn_freq = list(Counter(dataXn).values())  # True frequencies of the dataset Xn


dataTotal = np.concatenate((dataXs, dataXn))
Original_Freq_total = list(Counter(dataTotal).values())  # True frequencies of the dataset


epsilon = 3
d = 8

client_uoue = UOUEClient(epsilon=epsilon, Xs=dataXs, d=d)
server_uoue = UOUEServer(epsilon=epsilon, Xs=dataXs, d=d)
server_uoue_Xs = UOUEServer(epsilon=epsilon, Xs=dataXs, d=d)
server_uoue_Xn = UOUEServer(epsilon=epsilon, Xs=dataXs, d=d)

for item in dataTotal:
    # item is <class 'numpy.int32'>
    priv_uoue_data = client_uoue.privatise(item)
    server_uoue.aggregate(priv_uoue_data)
for item in dataXs:
    # item is <class 'numpy.int32'>
    priv_uoue_data = client_uoue.privatise(item)
    server_uoue_Xs.aggregate(priv_uoue_data)
for item in dataXn:
    # item is <class 'numpy.int32'>
    priv_uoue_data = client_uoue.privatise(item)
    server_uoue_Xn.aggregate(priv_uoue_data)

uoue_estimates = []
Xn_estimates = []
Xs_estimates = []

# For total
for i in range(0, d):
    uoue_estimates.append(round(server_uoue.estimate(i + 1)))
# For Xs
for i in range(0, dXs):
    Xs_estimates.append(round(server_uoue_Xs.estimate(i + 1)))

# For Xn
for i in range(0, dXn):
    Xn_estimates.append(round(server_uoue_Xn.estimate(i + 1)))

uoue_variance = 0
uoue_Xn_variance = 0
uoue_Xs_variance = 0


print("This is uoue_estimates size: \n" + str(len(uoue_estimates)))
print("This is Original_Freq_total size: \n" + str(len(uoue_estimates)))


for i in range(0, d):
    uoue_variance += (uoue_estimates[i] - Original_Freq_total[i]) ** 2
for i in range(0, dXs):
    uoue_Xs_variance += (Xs_estimates[i] - Original_Xs_freq[i]) ** 2
for i in range(0, dXn):
    uoue_Xn_variance += (Xn_estimates[i] - Original_Xn_freq[i]) ** 2

uoue_variance /= d
uoue_Xs_variance /= dXs
uoue_Xn_variance /= dXn

print("\n")
print("================================")
print("Size of Xs: ", len(dataXs), "Size of Xn ", len(dataXn), " with d=", d, " and epsilon= ", epsilon, "\n")
print("================================")

print("Utility-Optimised Unary Encoding (UOUE) Variance: ", uoue_variance)
print("Xs Variance: ", uoue_Xs_variance)
print("Xn Variance: ", uoue_Xn_variance)
