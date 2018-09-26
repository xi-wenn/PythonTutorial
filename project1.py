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
pp = pprint.PrettyPrinter(depth=200)

INPUT_FILE_NAME = 'Netflix_data.txt'
USER_COUNT = 0
SIGNATURE_MATRIX_ROWS = 1000
user_ratings = {} #key: user_id, value: list of movie_id(original)
movie_id_row_idx_map = {} #key: movie_id, value: row_index
f = open(INPUT_FILE_NAME, 'r')
current_movie_id = -1
movie_count = 0
user_ids_rated_over_20 = set()


################################## Part 1: generating matrix ###################################
for line in f:
  # if len(user_ratings) > 50: break
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
        user_ratings[user_id] = already_rated + [rating]




# print("before removal, user count = " + str(len(user_ratings)))
for user_id in user_ids_rated_over_20:
  user_ratings.pop(user_id, None)
# print("after removal, user count = " + str(len(user_ratings)))


# rating_lens = []
# for user, ratings in user_ratings.items():
#   rating_lens += [min(ratings)]

# rating_lens.sort()
# pp.pprint(rating_lens)

# print(len(user_ratings))
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

# # save user mapping to a file to use later
# with open('column_idx_user_id_map.pkl', 'wb') as f:
#   pickle.dump(column_idx_user_id_map, f, pickle.HIGHEST_PROTOCOL)

# # pdb.set_trace()

########################### Part3 : Data Structure Optimization ###############################
# movie_rating_sparse = csc_matrix(movie_rating_matrix)

compressed_movie_rating_matrix = np.zeros((20, USER_COUNT), dtype='int16')
user_counter = 0
for user_id, ratings in user_ratings.items():

  movie_counter = 0
  for movie_id in ratings:
    compressed_movie_rating_matrix[movie_counter, user_counter] = movie_id_row_idx_map[movie_id]
    # increment
    movie_counter += 1

  user_counter += 1

# #key: user_id, value:list of movie_id(continuous)
# user_ratings_with_mapped_movie_row_idx = {}
# # i = 0
# for user_id, ratings in user_ratings.items():
#   # pdb.set_trace()
#   temp_list = []
#   # user_id_column_idx_map[user_id] = i
#   # i += 1
#   for movie_id_before in ratings:
#     new_movie_id = movie_id_row_idx_map[movie_id_before]
#     temp_list.append(new_movie_id)
#   user_ratings_with_mapped_movie_row_idx[user_id] = temp_list

#user_ratings_with_mapped_movie_row_idx = collections.OrderedDict(user_ratings_with_mapped_movie_row_idx)

########################## part4 #############################################################

#the smallest prime number larger than 4499 (total number of movies)
R = 4507
signature_matrix = np.zeros((SIGNATURE_MATRIX_ROWS, USER_COUNT))
for i in range(SIGNATURE_MATRIX_ROWS):
  # first hash original 20 row matrix
  a = random.randint(0, R - 1)
  b = random.randint(0, R - 1)
  # hashed_movie_ratings = np.remainder((compressed_movie_rating_matrix * a + b), R)
  signature_matrix[i] = np.amin(np.remainder((compressed_movie_rating_matrix * a + b), R), axis = 0)

# pp.pprint(signature_matrix)

np.save('signature_matrix_file', signature_matrix)





# # First define some helper functions

# #f(x) = (ax +b) mod R
# def hash_func(a_val, b_val, x, R):
#   res = (a_val*x + b_val) % R
#   return res

# #get min_hashed value of a column
# def get_hashed_val(movie_row_idx_list, a_val, b_val, R):
#   min_val = R
#   for movie_row_idx in movie_row_idx_list:
#     val = hash_func(a_val,b_val,movie_row_idx,R)
#     if val < min_val:
#       min_val = val
#   return min_val

# #get 1 row of the signature matrix
# def get_sig_vec(user_ratings_after, a_val, b_val, R, USER_COUNT):
#   # pdb.set_trace()
#   vec = np.zeros(USER_COUNT)
#   for user_id, ratings in user_ratings_after.items():

#     vec[user_id_column_idx_map[user_id]] = get_hashed_val(ratings, a_val, b_val, R)
#   # pdb.set_trace()
#   return vec

# #get the signature matrix
# def get_sig_mat(a_list, b_list, user_ratings_after, USER_COUNT, R):
#   sig_mat = np.zeros([1000, USER_COUNT])
#   for i in range(1000):
#     # pdb.set_trace()
#     sig_mat[i,:] = get_sig_vec(user_ratings_after, a_list[i], b_list[i], R, USER_COUNT)
#   return sig_mat



# #the smallest prime number larger than 4499
# R = 4507
# #generate 1000 a_i, b_i, form the hash functions. a_i, b_i in [0, R-1]
# para_list = [i for i in range(4507)]
# a_list = random.sample(para_list, 1000)
# b_list = random.sample(para_list, 1000)

# signature_matrix = get_sig_mat(a_list, b_list, user_ratings_with_mapped_movie_row_idx, USER_COUNT, R)
# np.save('signature_matrix_file', signature_matrix)







# pp.pprint(user_ratings_with_mapped_movie_row_idx)
# 4507---prime number



# pp.pprint(movie_rating_matrix)

# print(movies)
# pp.pprint(user_ratings)
# print(len(user_ratings))



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









