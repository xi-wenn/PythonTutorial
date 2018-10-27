import numpy as np
import pprint
import random
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pdb
from scipy.sparse import csc_matrix
import collections
from tempfile import TemporaryFile
import itertools
import pickle


pp = pprint.PrettyPrinter(depth=200)

#--------------------------------------------------------------------------------------------
def pr(x, r, b):
  res =1 - (1 - (1-x)**r)**b
  return res

X = np.linspace(0,1,1000)
Y1 = pr(X, 100, 10)
Y2 = pr(X, 50, 20)
Y3 = pr(X, 25, 40)
Y4 = pr(X, 20, 50)
Y5 = pr(X, 12, 83.333)
Y6 = pr(X, 10, 100)
Y7 = pr(X, 11, 91)


plt.plot(X, Y1, label='r=100,b=10')
plt.plot(X, Y2, label='r=50,b=20')
plt.plot(X, Y3, label='r=25,b=40')
plt.plot(X, Y4, label='r=20,b=50')
plt.plot(X, Y5, label='r=12,b=83.333')
plt.plot(X, Y6, label='r=10,b=100')
plt.plot(X, Y7, label='r=11,b=91')
plt.vlines(0.35, 0, 1, colors = "c", linestyles = "dashed")
plt.xlabel('Distance')
plt.ylabel('Pr(hit)')
plt.legend()
plt.show()
