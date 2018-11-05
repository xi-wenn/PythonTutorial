from sklearn.datasets import fetch_rcv1
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation

import math
from sklearn.metrics import accuracy_score

######################################### Prep work ############################################
CCAT_ROW_IDX = 33
TRAIN_SIZE = 100000
rcv1 = fetch_rcv1()
num_rows, num_columns = rcv1.data.shape


############################ Problem 1a: generate CCAT vector ###################################
ccat_vector = np.full((num_rows), -1)

for i in range(num_rows):
  if rcv1.target[i, CCAT_ROW_IDX] == 1:
    ccat_vector[i] = 1


########################### Problem 1b: split training test set ###################################
data_train  = rcv1.data[:TRAIN_SIZE, :]
data_test   = rcv1.data[TRAIN_SIZE:, :]
label_train = ccat_vector[:TRAIN_SIZE]
label_test = ccat_vector[TRAIN_SIZE:]

################################### Problem 2: PEGASOS ############################################
max_iterations = 500
B = 200
lamda = 0.00025
pegasos_train_error = np.zeros([max_iterations+1, 1])
t = 1
#B is = number of points selected in the subset At
#def PEGASOS_SVM(data_train, label_train, max_iterations, lamda, B):

#initialize w = [0, ..., 0]
W_pegasos = np.zeros([data_train.shape[1], 1])

for t in range(1, max_iterations + 1):
#choose subset points At
    random_int_list = (np.random.randint(0, TRAIN_SIZE, 10*B))
    subset = data_train[random_int_list]@W_pegasos
    subset = ((np.diag(label_train[random_int_list]))@subset)
    indices = subset < 1
    indices = random_int_list.reshape(10*B, 1)[indices][:B]

    subgradient = lamda*W_pegasos - (np.sum((np.diag(label_train[indices]))@data_train[indices], axis = 0)).reshape(data_train.shape[1],1)/B
    step_size = 1 / (t * lamda)
    W_pegasos = W_pegasos - step_size * subgradient
    W_pegasos = W_pegasos * min(1, 1/(math.sqrt(lamda) * np.linalg.norm(W_pegasos)))

    #calculate the train_error at this iteration
    y_pred = data_train @ W_pegasos
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1
    pegasos_train_error[t] = 1 - accuracy_score(label_train, y_pred)
    print(pegasos_train_error[t], t)


plt.plot(range(1, max_iterations + 1), pegasos_train_error[1:])
plt.title("PEGASOS Traning Error")
plt.xlabel("Iterations")
plt.ylabel("Traning Error")
plt.show()


################################### Problem 3: AdaGrad ############################################

max_iterations = 500
B = 200
lamda = 0.00025
adagrad_train_error = np.zeros([max_iterations+1, 1])
t = 1
N = data_train.shape[1]
#B is = number of points selected in the subset At
#def PEGASOS_SVM(data_train, label_train, max_iterations, lamda, B):

#initialize W vector
W_agagrad = np.ones([data_train.shape[1], 1])/(np.sqrt(1/lamda))

eta = 1/(math.sqrt(max_iterations))

diagonal = np.zeros([data_train.shape[1], 1])



for t in range(1, max_iterations + 1):
    #choose subset points At
    random_int_list = (np.random.randint(0, TRAIN_SIZE, 10*B))
    subset = data_train[random_int_list]@W_agagrad
    subset = ((np.diag(label_train[random_int_list]))@subset)
    indices = subset < 1
    indices = random_int_list.reshape(10*B, 1)[indices]

    #update subgradient
    #gradient f_t(w_t)
    subgradient = lamda*W_agagrad - (np.sum((np.diag(label_train[indices]))@data_train[indices], axis = 0)).reshape(data_train.shape[1],1)/(indices.shape[0])


    if t == 1:
        W_agagrad = W_agagrad - eta * (subgradient)
    else:
        #pdb.set_trace()
        G = diagonal
        G = np.sqrt(G)
        G = 1/G #G_inverse
        step_size = 1 / (eta + math.sqrt(np.sum(np.absolute(subgradient))))

        #convert to diagnal matrix
        W_agagrad = W_agagrad - step_size * (G * subgradient)

    #projection
    W_agagrad = W_agagrad * min(1, 1/(math.sqrt(lamda) * np.linalg.norm(W_agagrad)))

    #update next Gt
    diagonal = diagonal + subgradient * subgradient

    #calculate the train_error at this iteration
    y_pred = data_train @ W_agagrad
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1

    #pdb.set_trace()

    adagrad_train_error[t] = 1 - accuracy_score(label_train, y_pred)
    print(adagrad_train_error[t], t)

plt.plot(range(1, 500 + 1), pegasos_train_error[1:])
plt.plot(range(1, 500 + 1), adagrad_train_error[1:])
plt.title("Traning Error")
plt.xlabel("Iterations")
plt.ylabel("Traning Error")
plt.show()


############################## Problem 4a: Neural Net; Keras #######################################
# Train neural nets over 5 epochs with 1, 2 and 3 layers, each with 100 hidden units. Plot the
# training error and include it in your report.

# first convert all labels from 1/-1 to 1/0
label_train_one_zero = (label_train + 1) / 2
label_test_one_zero = (label_test + 1) / 2

############ 1 hidden layer
model = Sequential([
    Dense(100, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(1),
    Activation('linear'),
])
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(data_train, label_train_one_zero, epochs=5, batch_size=128)
loss_and_metrics = model.evaluate(data_train, label_train_one_zero)
# print(loss_and_metrics)


############ 2 hidden layers
model_2 = Sequential([
    Dense(100, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(1),
    Activation('linear'),
])
model_2.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model_2.fit(data_train, label_train_one_zero, epochs=5, batch_size=128)
loss_and_metrics_2 = model_2.evaluate(data_train, label_train_one_zero)

############ 3 hidden layers
model_3 = Sequential([
    Dense(100, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(1),
    Activation('linear'),
])
model_3.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

model_3.fit(data_train, label_train_one_zero, epochs=5, batch_size=128)
loss_and_metrics_3 = model_3.evaluate(data_train, label_train_one_zero)


NN_training_errors = [1 - loss_and_metrics[1], 1 - loss_and_metrics_2[1], 1 - loss_and_metrics_3[1]]
plt.plot([1,2,3], NN_training_errors)
plt.title("Neural Network Traning Error (100 hidden units each layer)")
plt.xlabel("Layer Count")
plt.xticks([1,2,3])
plt.ylabel("Traning Error")
plt.show()


############################## Problem 4b: Neural Net; Keras #######################################
########### Test model 1
model_b1 = Sequential([
    Dense(50, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(1),
    Activation('linear')
])
model_b1.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model_b1.fit(data_train, label_train_one_zero, epochs=5, batch_size=128)
loss_and_metrics_b1 = model_b1.evaluate(data_train, label_train_one_zero)
print(loss_and_metrics_b1)

########### Test model 2
model_b2 = Sequential([
    Dense(50, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(1),
    Activation('linear')
])
model_b2.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model_b2.fit(data_train, label_train_one_zero, epochs=5, batch_size=128)
loss_and_metrics_b2 = model_b2.evaluate(data_train, label_train_one_zero)
print(loss_and_metrics_b2)

########### Test model 3
model_b3 = Sequential([
    Dense(50, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(1),
    Activation('linear')
])
model_b3.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model_b3.fit(data_train, label_train_one_zero, epochs=5, batch_size=128)
loss_and_metrics_b3 = model_b3.evaluate(data_train, label_train_one_zero)
print(loss_and_metrics_b3)

########### Test model 4
model_b4 = Sequential([
    Dense(50, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(50),
    Activation('relu'),
    Dense(1),
    Activation('linear')
])
model_b4.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model_b4.fit(data_train, label_train_one_zero, epochs=5, batch_size=128)
loss_and_metrics_b4 = model_b4.evaluate(data_train, label_train_one_zero)
print(loss_and_metrics_b4)

########### Test model 5
model_b5 = Sequential([
    Dense(50, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(100, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(100, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(50, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(1),
    Activation('linear')
])
model_b5.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model_b5.fit(data_train, label_train_one_zero, epochs=5, batch_size=128)
loss_and_metrics_b5 = model_b5.evaluate(data_train, label_train_one_zero)
print(loss_and_metrics_b5)

########### Test model 6
model_b6 = Sequential([
    Dense(50, input_shape=(num_columns,)),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(50),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(1),
    Activation('linear')
])
model_b6.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model_b6.fit(data_train, label_train_one_zero, epochs=5, batch_size=128)
loss_and_metrics_b6 = model_b6.evaluate(data_train, label_train_one_zero)
print(loss_and_metrics_b6)



############################## Problem 5: Test Results #######################################
# test PEGASOS
y_pred_test = data_test @ W_pegasos
y_pred_test[y_pred_test >= 0] = 1
y_pred_test[y_pred_test < 0] = -1
pegasos_test_error = 1 - accuracy_score(label_test, y_pred_test)
print(pegasos_test_error)

# test AdaGrad
y_pred_test = data_test @ W_agagrad
y_pred_test[y_pred_test >= 0] = 1
y_pred_test[y_pred_test < 0] = -1
adagrad_test_error = 1 - accuracy_score(label_test, y_pred_test)
print(adagrad_test_error )

# test neural net
test_loss_and_metrics_b6 = model_b6.evaluate(data_test, label_test_one_zero)
print(test_loss_and_metrics_b6)
