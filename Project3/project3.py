import numpy as np
import matplotlib.pyplot as plt
import sys

INPUT_FILE_NAME = 'yelp.csv'
f = open(INPUT_FILE_NAME, 'r', encoding="utf8")


data = []
############################ Part 1: Process data and formulate problem ###########################
isFirstLine = True
# k = 0
for line in f:
  if isFirstLine:
    # skip first line (headers)
    isFirstLine = False
#     k += 1
    continue

  data_row = line.strip().split(',')[2:] # strip off id and name
  # find where "yelping since" is (dedault idx 1), so we can handle cases where name includes comma
  data_start_idx = 0
  for i in range(1, len(data_row)):
    if data_row[i].find('-') > 0:
      break
    else:
      data_start_idx += 1
  data_row  = data_row[data_start_idx:]
  data_row[1] = data_row[1].replace('-', '') # format date as number
  if len(data_row) >  19:
    # has more than one elite year, then count how many there are, and use that as a feature
    years = data_row[6:-12]
    data_row = data_row[:6] + [len(years)] + data_row[-12:]
  else:
    data_row[6] = 0 if data_row[6] == 'None' else 1

  # map to float so we can process as numpy array
  data_row = list(map(float, data_row))

  if data == []:
    data = [data_row]
  else:
    data.append(data_row)
#   k += 1
#   if k % 10000 == 0:
#     print(k)
#     break

data = np.asarray(data)
print(data)