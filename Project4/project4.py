import numpy as np
import matplotlib.pyplot as plt
import sys

INPUT_FILE_NAME = 'yahoo_ad_clicks.csv'
f = open(INPUT_FILE_NAME, 'r', encoding="utf8")
data = []

for line in f:
  data.append(line.strip().split(','))

data = np.asarray(data, dtype='int8')