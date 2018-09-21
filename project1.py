import numpy as np
import pprint

pp = pprint.PrettyPrinter(depth=6)

INPUT_FILE_NAME = 'Netflix_data.txt'
user_ratings = {}
movies = []
f = open(INPUT_FILE_NAME, 'r')
i = 0
current_movie_id = -1
for line in f:
  if ':' in line:
    # is movie id
    # continue
    current_movie_id = line.rstrip(':\n')#.split(':')
    # movies += [current_movie_id]
    print(current_movie_id)
    # movies.append(movie_id)
  else:
    user_id, rating, date = line.split(',')
    # print(user_id, rating, date)
    user_id = int(user_id)
    rating = int(rating)

    if rating >= 3:
      rated_count = user_ratings,get(user_id, []).size()
      if rated_count >= 20
      user_ratings[user_id] = user_ratings.get(user_id, []) + [current_movie_id]

  i += 1

  if i >= 1000:
    break


print(movies)
pp.pprint(user_ratings)

