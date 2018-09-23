import numpy as np
import pprint
import random
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(depth=200)

INPUT_FILE_NAME = 'Netflix_data.txt'
user_ratings = {}
movies = {}
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


# pp.pprint(movie_rating_matrix)

# print(movies)
# pp.pprint(user_ratings)
# print(len(user_ratings))


def get_rand_user_pair(user_count):
  user_1 = -1
  user_2 = -1
  while user_1 == user_2:
    user_1 = random.randint(0, user_count)
    user_2 = random.randint(0, user_count)

  return frozenset({user_1, user_2})

############################# Part 2 : calculating jaccard distance ############################
NUM_PAIRS = 10000

i = 0
selected_pairs = set()
jaccard_distances = []
for i in range(NUM_PAIRS):
  user_pair = get_rand_user_pair(user_count)
  while user_pair in selected_pairs: # if already selected, re-draw
    user_pair = get_rand_user_pair(user_count)

  # unpack user values
  user_1, user_2 = user_pair
  user_1_data, user_2_data = movie_rating_matrix[:,user_1], movie_rating_matrix[:,user_2]
  intersection = np.sum(np.bitwise_and(user_1_data, user_2_data))
  union = np.sum(np.bitwise_or(user_1_data, user_2_data))
  jaccard_distances  += [1 - (intersection / union)]


num_bins = 50
print("Average distance = " + str(np.average(jaccard_distances)))
plt.hist(jaccard_distances, num_bins, facecolor='blue', alpha=0.5)
plt.show()









