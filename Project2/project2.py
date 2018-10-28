from sklearn.datasets import fetch_rcv1
from scipy import sparse
import numpy as np

######################################### Prep work ############################################
CCAT_ROW_IDX = 33
TRAIN_SIZE = 100000
rcv1 = fetch_rcv1()
num_rows, num_columns = rcv1.data.shape


############################ Problem 1a: generate CCAT vector ###################################
ccat_vector = np.full((num_rows), -1)

for i in range(num_rows):
  if rcv1.target[i, CCAT_ROW_IDX] == 1:
    ccat_vector[i] = 1

# print(ccat_vector)
# print(np.where(ccat_vector == 1)[0].size)


########################### Problem 1b: split training test set ###################################
data_train  = rcv1.data[:TRAIN_SIZE, :]
data_test   = rcv1.data[TRAIN_SIZE:, :]
label_train = ccat_vector[:TRAIN_SIZE]
label_test = ccat_vector[TRAIN_SIZE:]

# print(data_train)
# print(data_test)
# print(label_train)
# print(label_test)
