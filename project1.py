import numpy as np
import pprint

pp = pprint.PrettyPrinter(depth=200)

INPUT_FILE_NAME = 'Netflix_data.txt'
user_ratings = {}
movies = {}
f = open(INPUT_FILE_NAME, 'r')
current_movie_id = -1
movie_count = 0
user_ids_rated_over_20 = set()

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

print(len(user_ratings))

matrix = np.zeros( (movie_count, len(user_ratings)), dtype='int' )

user_idx = 0
for user_id, ratings in user_ratings.items():
  # print(user_id, ratings)
  for movie_id in ratings:
    matrix[movies[movie_id]][user_idx] = 1

  user_idx += 1


pp.pprint(matrix)


# print(movies)
# pp.pprint(user_ratings)
# print(len(user_ratings))
