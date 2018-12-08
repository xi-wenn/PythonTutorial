import numpy as np
import matplotlib.pyplot as plt
import sys

############################### initial file processing ###########################
INPUT_FILE_NAME = 'yahoo_ad_clicks.csv'
f = open(INPUT_FILE_NAME, 'r', encoding="utf8")
data = []

for line in f:
  data.append(line.strip().split(','))

data = np.asarray(data, dtype='int8')



################################ EXP3 #####################################
from numpy import random
n,T = data.shape
print(n, T)
pt = np.full((50,), 1.0/50)
Lt = np.zeros((50,))

# rgt = 0
# regret = []
choice = []
for t in range(1, T + 1):
  eta = np.sqrt(np.log(n)/(t*n))
  I = np.random.choice(50, 1, p=pt)[0]
  choice.append(I)
  lt = 1 - data[:, t-1]
  Lt += lt
  pt = np.exp(-eta*Lt) / np.sum(np.exp(-eta*Lt))

  # rgt += 1 - data[I][t-1]
  # regret.append(rgt/t)

plt.plot(regret)
plt.show()



def calc_regret(mu, max_mu, choice, T):
  R_t = []
  regret = 0
  for t in range(T):
    regret += max_mu - mu[choice[t]]
    R_t.append(regret / (t+1))
  return R_t

mu = np.sum(data, axis=1) / T
max_mu = np.max(mu)
print(mu)
print(max_mu)
print(max_mu*T)
# print(calc_regret(mu, max_mu, choice, T))
R = calc_regret(mu, max_mu, choice, T)
# print(R)
plt.plot(R)
plt.show()



