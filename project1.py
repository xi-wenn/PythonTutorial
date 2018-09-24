import numpy as np
import pprint
import random
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pdb 
from scipy.sparse import csc_matrix

pp = pprint.PrettyPrinter(depth=200)

INPUT_FILE_NAME = 'Netflix_data.txt'
user_ratings = {} #key: user_id, value: list of movie_id(original)
movies = {} #key: movie_id, value: row_index
f = open(INPUT_FILE_NAME, 'r')
current_movie_id = -1
movie_count = 0
user_ids_rated_over_20 = set()


################################## Part 1: generating matrix ###################################
for line in f:
  if ':' in line:
    # is movie id
    current_movie_id = int(line.rstrip(':\n'))
    movies[current_movie_id] = movie_count
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

# print(len(user_ratings))
user_count = len(user_ratings)
movie_rating_matrix = np.zeros( (movie_count, user_count), dtype='int' )

user_idx = 0
for user_id, ratings in user_ratings.items():
  # print(user_id, ratings)
  for movie_id in ratings:
    movie_rating_matrix[movies[movie_id]][user_idx] = 1

  user_idx += 1

########################### Part3 : Data Structure Optimization ###############################
# movie_rating_sparse = csc_matrix(movie_rating_matrix)

#key: user_id, value:list of movie_id(continuous)
user_ratings_mapped = {}
for user_id, ratings in user_ratings.items():
  temp_list = []
  for movie_id_before in ratings:
    new_movie_id = movies[movie_id_before]
    temp_list.append(new_movie_id)
  user_ratings_mapped[user_id] = temp_list

########################## part4 #############################################################
#the smallest prime number larger than 4499
R = 4507
#generate 1000 a_i, b_i, form the hash functions. a_i, b_i in [0, R-1]
para_list = [i for i in range(4507)]
a_list = random.sample(para_list, 1000)
b_list = random.sample(para_list, 1000)


#f(x) = (ax +b) mod R 
def hash_func(a, b, x, R):
  res = (a*x + b) % R
  return res

#get min_hashed value of a column
def get_hashed_val(movie_list, a, b, R):
  min_val = R
  for movie_id in movie_list:
    val = hash_func(a,b,movie_id,R)
    if val < min_val:
      min_val = val
  return min_val

#get 1 row of the signature matrix
def get_sig_vec(user_ratings_mapped, a, b, R, user_count):
  vec = np.zeros(user_count)
  i = 0
  for user_id, ratings in user_ratings.items():
    vec[i] = get_hashed_val(user_ratings_mapped[user_id], a, b, R)
    i += 1
  return vec

#get the signature matrix
def get_sig_mat(a_list, b_list, user_ratings_mapped, user_count,R):
  sig_mat = np.zeros([1000, user_count])
  for i in range(1000):
    a = a_list[i]
    b = b_list[i]
    sig_mat[i,:] = get_sig_vec(user_ratings_mapped, a, b, R, user_count)
  return sig_mat

signature_matrix = get_sig_vec(a_list, b_list, user_ratings_mapped, user_count,R)

  



#pp.pprint(user_ratings_mapped)
#4507---prime number



# pp.pprint(movie_rating_matrix)

# print(movies)
# pp.pprint(user_ratings)
# print(len(user_ratings))



############################# Part 2 : calculating jaccard distance ############################
# def get_rand_user_pair(user_count):
#   user_1 = -1
#   user_2 = -1
#   while user_1 == user_2:
#     user_1 = random.randint(0, user_count)
#     user_2 = random.randint(0, user_count)

#   return frozenset({user_1, user_2})

# NUM_PAIRS = 10000

# i = 0
# selected_pairs = set()
# jaccard_distances = []
# for i in range(NUM_PAIRS):
#   user_pair = get_rand_user_pair(user_count)
#   while user_pair in selected_pairs: # if already selected, re-draw
#     user_pair = get_rand_user_pair(user_count)

#   # unpack user values
#   user_1, user_2 d= user_pair
#   user_1_data, user_2_data = movie_rating_matrix[:,user_1], movie_rating_matrix[:,user_2]
#   intersection = np.sum(np.bitwise_and(user_1_data, user_2_data))
#   union = np.sum(np.bitwise_or(user_1_data, user_2_data))
#   jaccard_distances  += [1 - (intersection / union)]


# num_bins = 50
# print("Average distance = " + str(np.average(jaccard_distances)))
# plt.hist(jaccard_distances, num_bins, facecolor='blue', alpha=0.5)
# plt.show()









