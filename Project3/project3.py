import numpy as np
import matplotlib.pyplot as plt
import sys

INPUT_FILE_NAME = 'yelp.csv'
f = open(INPUT_FILE_NAME, 'r')


data = None
############################ Part 1: Process data and formulate problem ###########################
isFirstLine = True
# i = 0
for line in f:

  if isFirstLine:
    # skip first line (headers)
    isFirstLine = False
    continue

  data_row = line.strip().split(',')[2:] # strip off id and name
  data_row[1] = data_row[1].replace('-', '') # format date as number
  if len(data_row) >  19:
    # has more than one elite year, then count how many there are, and use that as a feature
    years = data_row[6:-12]
    data_row = data_row[:6] + [len(years)] + data_row[-12:]
  else:
    data_row[6] = 0 if data_row[6] == 'None' else 1

  # map to float so we can process as numpy array
  data_row = list(map(float, data_row))
  if data is None:
    data = np.asarray(data_row)
  else:
    data = np.vstack((data, np.asarray(data_row)))
  # data.append(data_row)
#   i += 1
#   if i > 100 :
#     print(i)
#     break

# print(data)

