import numpy as np
import pprint
import random
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pdb
from scipy.sparse import csc_matrix
import collections
from tempfile import TemporaryFile
import pickle
import sys
import itertools

pp = pprint.PrettyPrinter(depth=None)

INPUT_FILE_NAME = 'Netflix_data.txt'
USER_COUNT = 0
SIGNATURE_MATRIX_ROWS = 1001
user_ratings = {} #key: user_id, value: list of movie_id(original)
movie_id_row_idx_map = {} #key: movie_id, value: row_index
f = open(INPUT_FILE_NAME, 'r')
current_movie_id = -1
movie_count = 0
user_ids_rated_over_20 = set()


################################## Part 1: generating matrix ###################################
for line in f:
  if ':' in line:
    # is movie id
    current_movie_id = int(line.rstrip(':\n'))
    movie_id_row_idx_map[current_movie_id] = movie_count
    movie_count += 1
  else:
    user_id, rating, date = line.split(',')
    user_id = int(user_id)
    rating = int(rating)

    if rating >= 3:
      already_rated = user_ratings.get(user_id, [])

      if already_rated == -1 or len(already_rated) >= 20:
        # over 20 movies rated, ignore this user
        user_ratings[user_id] = -1
        user_ids_rated_over_20.add(user_id)
        continue
      else:
        # under 20 rated
        user_ratings[user_id] = already_rated + [current_movie_id]

for user_id in user_ids_rated_over_20:
  user_ratings.pop(user_id, None)

USER_COUNT = len(user_ratings)
movie_rating_matrix = np.zeros( (movie_count, USER_COUNT), dtype='int8' )

user_id_column_idx_map = {} # key: user_id, value: index of user_id; to be used in later parts
column_idx_user_id_map = {}
user_idx = 0
for user_id, ratings in user_ratings.items():
  # print(user_id, ratings)
  for movie_id in ratings:
    movie_rating_matrix[movie_id_row_idx_map[movie_id]][user_idx] = 1
  user_id_column_idx_map[user_id] = user_idx
  column_idx_user_id_map[user_idx] = user_id
  user_idx += 1



############################# Part 2 : calculating jaccard distance ############################
# def get_rand_user_pair(USER_COUNT):
#   user_1 = -1
#   user_2 = -1
#   while user_1 == user_2:
#     user_1 = random.randint(0, USER_COUNT - 1)
#     user_2 = random.randint(0, USER_COUNT - 1)

#   return frozenset({user_1, user_2})

# NUM_PAIRS = 10000

# i = 0
# selected_pairs = set()
# jaccard_distances = []
# for i in range(NUM_PAIRS):
#   user_pair = get_rand_user_pair(USER_COUNT)
#   while user_pair in selected_pairs: # if already selected, re-draw
#     user_pair = get_rand_user_pair(USER_COUNT)

#   # unpack user values
#   user_1, user_2 = user_pair
#   user_1_data, user_2_data = movie_rating_matrix[:,user_1], movie_rating_matrix[:,user_2]
#   intersection = np.sum(np.bitwise_and(user_1_data, user_2_data))
#   union = np.sum(np.bitwise_or(user_1_data, user_2_data))
#   jaccard_distances  += [1 - (intersection / union)]


# num_bins = 50
# print("Average distance = " + str(np.average(jaccard_distances)))
# print("Lowest distance = " + str(np.amin(jaccard_distances)))
# plt.hist(jaccard_distances, num_bins, facecolor='blue', alpha=0.5)
# plt.title("Jaccard Distance of 10,000 Random User Pairs")
# plt.xlabel("Jaccard Distance")
# plt.ylabel("User Pair Count")
# plt.show()


########################### Part3 : Data Structure Optimization ###############################
compressed_movie_rating_matrix = np.zeros((20, USER_COUNT), dtype='int16')

user_counter = 0
for user_id, ratings in user_ratings.items():
  # loop through movies rated by a user
  movie_counter = 0
  for movie_id in ratings:
    compressed_movie_rating_matrix[movie_counter, user_counter] = movie_id_row_idx_map[movie_id]
    movie_counter += 1

  user_counter += 1

# doesn't need original matrix from now on, delete to save ram
del movie_rating_matrix




########################## part4 #############################################################

#the smallest prime number larger than 4499 (total number of movies)
R = 4507
signature_matrix = np.zeros((SIGNATURE_MATRIX_ROWS, USER_COUNT))
for i in range(SIGNATURE_MATRIX_ROWS):
  # first hash original 20 row matrix
  a = random.randint(0, R - 1)
  b = random.randint(0, R - 1)
  signature_matrix[i] = np.amin(np.remainder((compressed_movie_rating_matrix * a + b), R), axis = 0)

pdb.set_trace()

r = 11
b = 91
P = 45491

# def remove_single_appearance_values(vals, count):
#   res = []
#   for i in range(len(count)):
#     if count[i] > 1:
#       res.append(vals[i])
#   return res

# def map_to_buckets(i, x, buckets):
#   # if x in repeated_vals:
#   buckets[x] = buckets.get(x, []) + [i]



close_user_pairs = set()
for band_index in range(b):
  a = np.diag(np.random.choice(P, r))
  b = np.random.choice(P, r).reshape((r, 1))
  cur_band_matrix = signature_matrix[band_index * r : (band_index + 1) * r, :]

  res_mat = (a @ cur_band_matrix + b) % P
  val_list = np.sum(res_mat, axis = 0)
  # vals, count = np.unique(val_list, return_counts=True)
  # repeated_val = remove_single_appearance_values(vals, count)


  buckets = {}
  # map(lambda i, x, buckets = buckets: buckets[x] = buckets.get(x, []) + [i], enumerate(val_list))
  for idx, value in enumerate(val_list):
    # if value in repeated_val:
      # pdb.set_trace()
    buckets[(band_index, value)] = buckets.get(value, []) + [idx]

  pdb.set_trace()
  for bucket_key, bucket_values in buckets.items():
    if len(bucket_values) > 1:
      for pair in itertools.combinations(bucket_values, 2):
        # col_idx_1, col_idx_2 = pair
        # temp_list = []
        # temp_list.append(column_idx_user_id_map[col_idx_1])
        # temp_list.append(column_idx_user_id_map[col_idx_2])
        # close_user_pairs.add(frozenset( temp_list))
        close_user_pairs.add(frozenset(pair))








