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

################################ Thompson ########################################
#Thompson
def Thompson(data):
    S = np.zeros([data.shape[0], 1]) + 100
    F = np.zeros([data.shape[0], 1]) + 11000
    T = data.shape[1]

    reward = []
    choice = []
    rw = 0
    u_dict = {}

    for t in range(1, T+1):
        theta = np.random.beta(S+1,F+1)
        i = np.argmax(theta)
        u_dict[i] = u_dict.get(i, []) + [data[i][t-1]]
        choice.append(i)
        if data[i][t-1] == 1:
            S[i] += 1
            rw += 1
        else:
            S[i] *= 0.98
            F[i] += 1
        reward.append(rw)

    #calculate regret
    mean = np.zeros([data.shape[0], 1])
    for key, val in u_dict.items():
        mean[key] = sum(val)/len(val)

    best_mu = np.max(mean)
    regret = []
    rgt = 0
    for i in range(T):
        rgt += best_mu - mean[choice[i]]
        regret.append(rgt/(i+1))

    plt.plot(range(0,len(regret)), regret)
    plt.title("Regret over time using Thompson Sampling (modified version)")
    plt.show()
    print("Total regret with Thompson:", regret[-1])
    plt.title("Reward over time using Thompson Sampling (modified version)")
    plt.plot(range(0,len(reward)), reward)
    plt.show()
    print("Total reward with Thompson:", reward[-1])

Thompson(data)


################################ UCB ######################################
def UCB(data):
    Mu = np.zeros([data.shape[0], 1])
    Ni = np.ones([data.shape[0], 1])
    T = data.shape[1]
    for t in range(1, data.shape[0] + 1):
        if data[t-1][t-1] == 1:
            Mu[t-1] = 1

    regret = []
    reward = []
    choice = []
    rgt = 0
    rw = 0
    u_dict = {}
    for t in range(data.shape[0] + 1, T + 1):
        UCB = Mu + np.sqrt(2*np.log(t - data.shape[0])/Ni)
        j = np.argmax(UCB)
        yt = data[j][t-1]
        Ni[j] += 1
        Mu[j] += (yt - Mu[j])/Ni[j]
        selected = np.argmax(Mu);
        choice.append(selected)
        u_dict[selected] = u_dict.get(selected, []) + [data[selected][t-1]]
        rw += data[selected][t-1]
        reward.append(rw)

    #calculate the regret
    mean = np.zeros([data.shape[0], 1])
    for key, val in u_dict.items():
        mean[key] = sum(val)/len(val)

    best_mu = np.max(mean)
    regret = []
    rgt = 0
    for i in range(T-50):
        rgt += best_mu - mean[choice[i]]
        regret.append(rgt/(i+1))
    plt.plot(range(0,len(regret)), regret)
    plt.show()
    plt.plot(range(0,len(reward)), reward)
    plt.show()
    print("total reward with UCB:", reward[-1])

UCB(data)



################################ EXP3 #####################################
from numpy import random
def exp_three(data):
    n,T = data.shape
    pt = np.full((n,), 1.0/50)
    Lt = np.zeros((n,))

    reward_list = []
    reward = 0
    choice = []
    for t in range(1, T + 1):
        eta = np.sqrt(np.log(n)/(t*n))
        I = np.random.choice(n, 1, p=pt)[0]
        choice.append(I)
        lt = 1 - data[:, t-1]
        Lt += lt
        pt = np.exp(-eta*Lt) / np.sum(np.exp(-eta*Lt))

        reward += data[I][t-1]
        reward_list.append(reward)

    return (choice, reward_list)

def calc_regret(mu, max_mu, choice, T):
    R_t = []
    regret = 0
    for t in range(T):
        regret += max_mu - mu[choice[t]]
        R_t.append(regret / (t+1))
    return R_t

choice, reward_list = exp_three(data)

mu = np.sum(data, axis=1) / T
max_mu = np.max(mu)

R = calc_regret(mu, max_mu, choice, T)

print("Total regret with EXP3 =", R[-1])
plt.title("Regret over time using EXP3 (full feedback)")
plt.plot(R)
plt.show()

print("Total reward with EXP3 =", reward_list[-1])
plt.title("Reward over time using EXP3 (full feedback)")
plt.plot(reward_list)
plt.show()


