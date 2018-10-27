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
def get_rand_user_pair(USER_COUNT):
  user_1 = -1
  user_2 = -1
  while user_1 == user_2:
    user_1 = random.randint(0, USER_COUNT - 1)
    user_2 = random.randint(0, USER_COUNT - 1)

  return frozenset({user_1, user_2})

NUM_PAIRS = 10000

i = 0
selected_pairs = set()
jaccard_distances = []
for i in range(NUM_PAIRS):
  user_pair = get_rand_user_pair(USER_COUNT)
  while user_pair in selected_pairs: # if already selected, re-draw
    user_pair = get_rand_user_pair(USER_COUNT)

  # unpack user values
  user_1, user_2 = user_pair
  user_1_data, user_2_data = movie_rating_matrix[:,user_1], movie_rating_matrix[:,user_2]
  intersection = np.sum(np.bitwise_and(user_1_data, user_2_data))
  union = np.sum(np.bitwise_or(user_1_data, user_2_data))
  jaccard_distances  += [1 - (intersection / union)]


num_bins = 50
print("Average distance = " + str(np.average(jaccard_distances)))
print("Lowest distance = " + str(np.amin(jaccard_distances)))
plt.hist(jaccard_distances, num_bins, facecolor='blue', alpha=0.5)
plt.title("Jaccard Distance of 10,000 Random User Pairs")
plt.xlabel("Jaccard Distance")
plt.ylabel("User Pair Count")
plt.show()


########################### Part3 : Data Structure Optimization ###############################
compressed_movie_rating_matrix = np.zeros((20, USER_COUNT), dtype='int16')

user_counter = 0
for user_id, ratings in user_ratings.items():
  # loop through movies rated by a user
  movie_counter = 0
  for movie_id in ratings:
    # add one to row idx since 0 is a valid movie entry
    compressed_movie_rating_matrix[movie_counter, user_counter] = movie_id_row_idx_map[movie_id] + 1
    movie_counter += 1

  user_counter += 1

# doesn't need original matrix from now on, delete to save ram
del movie_rating_matrix




########################## part4 #############################################################

#the smallest prime number larger than 4499 (total number of movies)
R = 4507
a_list = np.random.choice(R, SIGNATURE_MATRIX_ROWS)
b_list = np.random.choice(R, SIGNATURE_MATRIX_ROWS)
signature_matrix = np.zeros((SIGNATURE_MATRIX_ROWS, USER_COUNT))
for i in range(SIGNATURE_MATRIX_ROWS):
  # first hash original 20 row matrix
  a = a_list[i]
  b = b_list[i]
  signature_matrix[i] = np.amin(np.remainder((compressed_movie_rating_matrix * a + b), R), axis = 0)


r = 11
band_num = 91
P = 45491

actual_close_pairs = set()
a_diag = np.diag(np.random.choice(P, r))
b_col = np.random.choice(P, r).reshape((r, 1))
list_buckets = []
for band_index in range(band_num):
  cur_band_matrix = signature_matrix[band_index * r : (band_index + 1) * r, :]
  res_mat = (a_diag @ cur_band_matrix + b_col) % P
  val_list = np.sum(res_mat, axis = 0)

  buckets = {}

  for idx, value in enumerate(val_list):
    buckets[value] = buckets.get((value), []) + [idx]
  list_buckets.append(buckets)

  for bucket_key, bucket_values in buckets.items():
    if len(bucket_values) > 1:
      for pair in itertools.combinations(bucket_values, 2):
        user_1, user_2 = pair
        pair_set = frozenset(pair)

        # if already seen pair before, continue
        if (pair_set in actual_close_pairs):
          continue

        # else calculate jaccard distance and add to appropriate set
        user_1_data, user_2_data = compressed_movie_rating_matrix[:,user_1], compressed_movie_rating_matrix[:,user_2]
        intersection = len(np.intersect1d(user_1_data[user_1_data > 0], user_2_data[user_2_data > 0]))
        union = np.count_nonzero(np.unique(np.append(user_1_data, user_2_data)))

        if union == 0:
          print(intersection)
          print(np.unique(user_1_data + user_2_data))
          print(union)
        if (1 - (intersection / union)) < 0.35:
          actual_close_pairs.add(pair_set)


  # print(band_index)
  # print(len(false_positives))
  # print(len(actual_close_pairs))
  # print(close_user_pairs)

######################### Part 5 : nearest neighbors #######################################################
def find_nearest_user(new_movie_list, list_buckets):

  #map the movie_id in the given list to 1 - 4499 by above dictionary
  compressed_movie_rating_vec = np.zeros(20, dtype='int16')
  row_count = 0
  for movie_id in new_movie_list:
    compressed_movie_rating_vec[row_count] = movie_id_row_idx_map[movie_id] + 1
    row_count += 1

  #get the 1001 by 1 signature vector
  signature_vec = np.zeros(SIGNATURE_MATRIX_ROWS)
  for i in range(SIGNATURE_MATRIX_ROWS):
    a = a_list[i]
    b = b_list[i]
    signature_vec[i] = np.amin(np.remainder((compressed_movie_rating_vec * a + b), R))

  #hash the signature vector to get the bucket values
  bucket_dict_new = {}
  for band_idx in range(band_num):
    cur_band_vec = signature_vec[band_idx * r : (band_idx + 1) * r]
    res_mat = (a_diag @ cur_band_vec.reshape((r,1)) + b_col) % P
    val = np.sum(res_mat, axis = 0)
    bucket_dict_new[band_idx] = val

  nearest_set = set()
  #loop all the bucket values, find nearest neighbor
  for band_idx, val in bucket_dict_new.items():
    #get the related set of similar movies
    user_list = (list_buckets[band_idx]).get(int(val), [])
    if len(user_list) == 0: continue
    for user_id in user_list:
      user_1_data , user_2_data = compressed_movie_rating_vec, compressed_movie_rating_matrix[:,user_id]
      intersection = len(np.intersect1d(user_1_data[user_1_data > 0], user_2_data[user_2_data > 0]))
      union = np.count_nonzero(np.unique(np.append(user_1_data, user_2_data)))
      jaccard_dist = 1 - (intersection / union)
      if jaccard_dist < 0.35:
        nearest_set.add(column_idx_user_id_map[user_id])
    pdb.set_trace()

  return nearest_set

new_movie_list = [178, 761, 4432]
nearest_user = find_nearest_user(new_movie_list, list_buckets)
print(nearest_user)





























