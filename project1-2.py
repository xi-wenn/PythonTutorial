import numpy as np
import pprint
import random
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pdb
from scipy.sparse import csc_matrix
import collections
from tempfile import TemporaryFile

pp = pprint.PrettyPrinter(depth=200)

def pr(x, r, b):
  res =1 - (1 - (1-x)**r)**b
  return res

#xpdb.set_trace()

X = np.linspace(0,1,1000)
Y1 = pr(X, 100, 10)
Y2 = pr(X, 50, 20)
Y3 = pr(X, 25, 40)
Y4 = pr(X, 20, 50)
Y5 = pr(X, 12, 83.333)
Y6 = pr(X, 10, 100)

plt.plot(X, Y1, label='r=100,b=10')
plt.plot(X, Y2, label='r=50,b=20')
plt.plot(X, Y3, label='r=25,b=40')
plt.plot(X, Y4, label='r=20,b=50')
plt.plot(X, Y5, label='r=12,b=83.333')
plt.plot(X, Y6, label='r=10,b=100')
plt.vlines(0.35, 0, 1, colors = "c", linestyles = "dashed")
plt.xlabel('Distance')
plt.ylabel('Pr(hit)')
plt.legend()
plt.show()


#we choose r = 10, b = 100
signature_mat = np.load('signature_matrix_file.npy')
#pp.pprint(signature_mat[0,:])

r = 10
b = 100

#generate 10 hash functions h_i(x) = (a_i * x + b_i) mod P
P = 1019
para_list = [i for i in range(P)]
a_list = random.sample(para_list, 10)
b_list = random.sample(para_list, 10)

#given a vector v, get h(v)
def hash_vector(vec, a_list, b_list, P):
  res = 0
  for i in range(len(vec)):
    res += (a_list[i] * vec[i] + b_list[i]) mod P
  return res

#check if two column are similar (dist < 0.35)
def is_similar(vec_a, vec_b, a_list, b_list, P):
  hashed_val1 = hash_vector(vec_a, a_list, b_list, P)
  hashed_val2 = hash_vector(vec_b, a_list, b_list, P)
  if hashed_val1 == hashed_val2:
    return True
  return False




























