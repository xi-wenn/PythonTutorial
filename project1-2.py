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

def pr(x, r, b):
  res =1 - (1 - (1-x)**r)**b
  return res

#xpdb.set_trace()

X = np.linspace(0,1,1000)
# Y1 = pr(X, 100, 10)
# Y2 = pr(X, 50, 20)
# Y3 = pr(X, 25, 40)
# Y4 = pr(X, 20, 50)
# Y5 = pr(X, 12, 83.333)
# Y6 = pr(X, 10, 100)
Y7 = pr(X, 11, 100)
Y8 = pr(X, 20, 100)

# plt.plot(X, Y1, label='r=100,b=10')
# plt.plot(X, Y2, label='r=50,b=20')
# plt.plot(X, Y3, label='r=25,b=40')
# plt.plot(X, Y4, label='r=20,b=50')
# plt.plot(X, Y5, label='r=12,b=83.333')
# plt.plot(X, Y6, label='r=10,b=100')
plt.plot(X, Y7, label='r=11,b=91')
plt.plot(X, Y8, label='r=20,b=100')
plt.vlines(0.35, 0, 1, colors = "c", linestyles = "dashed")
plt.xlabel('Distance')
plt.ylabel('Pr(hit)')
plt.legend()
plt.show()


# #we choose r = 10, b = 100
# signature_mat = np.load('signature_matrix_file.npy')
# #pp.pprint(signature_mat[0,:])

# r = 10
# b = 100


# ################################# new version ###########################

# P = 45491

# #remove the values that only appears once
# def remove_single_appearance_values(vals, count):
#   res = []
#   for i in range(len(count)):
#     if count[i] > 1:
#       res.append(vals[i])
#   return res


# close_user_pairs = set()
# for band_index in range(b):
#   a = np.diag(np.random.choice(P, r))
#   b = np.random.choice(P, r).reshape((r, 1))
#   cur_band_matrix = signature_mat[band_index * r : (band_index + 1) * r, :]

#   # get_ith_band(signature_mat, band_index, r)


#   # diag_mat = np.diag(a)
#   # B_mat = generate_matrix(b, signature_mat.shape[1])
#   res_mat = (a @ cur_band_matrix + b) % P
#   val_list = np.sum(res_mat, axis = 0)
#   vals, count = np.unique(val_list, return_counts=True)
#   repeated_val = remove_single_appearance_values(vals, count)

#   buckets = {} #dict.fromkeys(repeated_val, [])
#   for idx, value in enumerate(val_list):
#     if value in repeated_val:
#       # pdb.set_trace()
#       buckets[value] = buckets.get(value, []) + [idx]

#   pdb.set_trace()
#   for bucket_key, bucket_values in buckets.items():
#     for pair in itertools.combinations(bucket_values, 2):
#       # col_idx_1, col_idx_2 = pair
#       # temp_list = []
#       # temp_list.append(column_idx_user_id_map[col_idx_1])
#       # temp_list.append(column_idx_user_id_map[col_idx_2])
#       # close_user_pairs.add(frozenset( temp_list))
#       close_user_pairs.add(frozenset(pair))



############################### end new version #########################















# #generate 10 hash functions h_i(x) = (a_i * x + b_i) mod P
# P = 42709
# para_list = [i for i in range(P)]
# a_list = random.sample(para_list, r)
# b_list = random.sample(para_list, r)

# #given a vector v, get h(v)
# def hash_vector(vec, a_list, b_list, P):
#   res = 0
#   for i in range(len(vec)):
#     res += (a_list[i] * vec[i] + b_list[i]) % P
#   return res

# #check if two column are similar (dist < 0.35)
# def is_equal(vec_a, vec_b, a_list, b_list, P):
#   hashed_val1 = hash_vector(vec_a, a_list, b_list, P)
#   hashed_val2 = hash_vector(vec_b, a_list, b_list, P)
#   if hashed_val1 == hashed_val2:
#     return True
#   return False

# def get_ith_band(signature_mat, i, r):
#   return signature_mat[i*r:i*r+r, :]

# def generate_matrix(b_list, N):
#   T = []
#   for i in range(len(b_list)):
#     row = [b_list[i]] * N
#     T += row
#   T = np.array(T)
#   T = T.reshape(len(b_list), N)
#   return T

# #remove the values that only appears once
# def remove_single_appearance_values(vals, count):
#   res = []
#   for i in range(len(count)):
#     if count[i] > 1:
#       res.append(vals[i])
#   return res


# # load user id mapings
# column_idx_user_id_map = {}
# with open('column_idx_user_id_map.pkl', 'rb') as f:
#   column_idx_user_id_map = pickle.load(f)

# close_user_pairs = set()
# pdb.set_trace()

# for band_index in range(b):
#   cur_band_matrix = get_ith_band(signature_mat, band_index, r)
#   diag_mat = np.diag(a_list)
#   B_mat = generate_matrix(b_list, signature_mat.shape[1])
#   res_mat = (diag_mat @ cur_band_matrix + B_mat) % P
#   val_list = np.sum(res_mat, axis = 0)
#   vals, count = np.unique(val_list, return_counts=True)
#   repeated_val = remove_single_appearance_values(vals, count)

#   buckets = {} #dict.fromkeys(repeated_val, [])
#   for idx, value in enumerate(val_list):
#     if value in repeated_val:
#       # pdb.set_trace()
#       buckets[value] = buckets.get(value, []) + [idx]

#   pdb.set_trace()
#   for bucket_key, bucket_values in buckets.items():
#     for pair in itertools.combinations(bucket_values, 2):
#       # col_idx_1, col_idx_2 = pair
#       # temp_list = []
#       # temp_list.append(column_idx_user_id_map[col_idx_1])
#       # temp_list.append(column_idx_user_id_map[col_idx_2])
#       # close_user_pairs.add(frozenset( temp_list))
#       close_user_pairs.add(frozenset(pair))

# pdb.set_trace()
# pp.pprint(close_user_pairs)



# def load_obj(name ):
#     with open('obj/' + name + '.pkl', 'rb') as f:
#         return pickle.load(f)






