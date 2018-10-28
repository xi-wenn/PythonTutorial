from sklearn.datasets import fetch_rcv1
from scipy import sparse
import numpy as np

CCAT_ROW_IDX = 33


rcv1 = fetch_rcv1()
num_rows, num_columns = rcv1.data.shape
# print(num_rows)
# print(num_columns)

ccat_vector = np.full((num_rows), -1)

for i in range(num_rows):
  if rcv1.target[i, CCAT_ROW_IDX] == 1:
    ccat_vector[i] = 1

# print(ccat_vector)
# print(np.where(ccat_vector == 1)[0].size)

# print(np.where(rcv1.target_names == 'CCAT'))

# print((0, 700) in get_items(rcv1['data']))
# print((0, 2292) in get_items(rcv1['data']))
# print((0, 2) in get_items(rcv1['target']))
# print((0, 93) in get_items(rcv1['target']))

# print(rcv1['data'][0][700])
# print(rcv1['data'][0][2292])
# print(rcv1['target'][0][2])
# print(rcv1['target'][0][93])

# print(rcv1['data'])
# print(rcv1['target'])