import numpy as np
import pprint

pp = pprint.PrettyPrinter(depth=6)

INPUT_FILE_NAME = 'Netflix_data.txt'
user_ratings = {}
movies = {}
f = open(INPUT_FILE_NAME, 'r')
# i = 0
current_movie_id = -1
movie_count = 0
user_ids_rated_over_20 = set()

for line in f:
  if ':' in line:
    # is movie id
    current_movie_id = int(line.rstrip(':\n')) #.split(':')
    movie_count += 1
    # movies += [current_movie_id]
    # print(current_movie_id)
    movies[current_movie_id] = movie_count
  else:
    user_id, rating, date = line.split(',')
    # print(user_id, rating, date)
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

  # i += 1
  # if i >= 1000:
  #   break

# print(len(movies))
# print(len(user_ratings))
# print(movies)

for user_id in user_ids_rated_over_20:
  user_ratings.pop(user_id, None)

print(len(user_ratings))

# matrix = np.zeros( (movie_count, len(user_ratings)) )

# i = 0
# for user_id, ratings in user_ratings.items():
#   # print(user_id, ratings)
#   for movie_id in ratings:
#     matrix[movies[movie_id][i]] = 1

#   i += 1


# pp.pprint(matrix)


# print(movies)
pp.pprint(user_ratings)
# print(len(user_ratings))
