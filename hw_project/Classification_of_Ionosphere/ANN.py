#Setting up the required imports and extensions
#that will be used to construct this ANN
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime, timedelta
from collections import deque
global sum_mse
sum_mse = 0
dataframe = pd.read_csv('ionosphere.csv')
dataframe.describe()
dataframe.tail(10)

desired_columns = ['class' , 'a02']
change_data= dataframe[desired_columns]
number_of = len(dataframe[desired_columns])
print(type(change_data['class'][0]))

for i in range(number_of):
    if change_data['class'][i] == 'g':
        change_data['class'][i] = 1
    elif change_data['class'][i] == 'b':
        change_data['class'][i] = 0
print(dataframe[desired_columns])
basic_mlp_data = change_data
basic_mlp_data.head()

amountEpochs = 1000
batchSize = 128
netHiddenSizes = [128, 64, 8]
learningRate = 0.001
strength = 0.01
nonLinearity = tf.nn.relu
dropoutAmount = 0.7

netInput = tf.placeholder(tf.float32, shape= [None, 1])
netTarget = tf.placeholder(tf.float32, shape= [None, 1])
dropoutProb = tf.placeholder(tf.float32)


regulariser = tf.contrib.layers.l2_regularizer(scale=strength)

net = netInput
for size in netHiddenSizes:
    net = tf.layers.dense(inputs = net,
                          units = size,
                          activation = nonLinearity,
                          kernel_regularizer = regulariser)

    net = tf.layers.dropout(inputs = net,
                            rate = dropoutProb)

#This simple MLP will produce a linear output value
netOutput = tf.layers.dense(inputs = net,
                            units = 1,
                            activation = None,
                            kernel_regularizer = regulariser)

#The loss that is determined for punishing the network
#based on how efficient it is (MSE)
loss = tf.losses.mean_squared_error(labels = netTarget,
                                    predictions = netOutput)

#Applying a L2_loss to the current loss using Tensorflow
l2Variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
l2Loss = tf.contrib.layers.apply_regularization(regulariser,
                                                 l2Variables)
totalLoss = loss + l2Loss

#Starts the training of the ANN with the initialzation of the
#Tensorflow operations that will be required
trainOp = tf.train.AdamOptimizer(learningRate).minimize(totalLoss)
initOp = tf.global_variables_initializer()

with tf.Session() as sess:

#Sets up the amount of folds that will be used as well as
#the fold error and iteration DataFrame.to_numpy()
    amountFolds = 10
    kFolds = KFold(n_splits=amountFolds)
    data = basic_mlp_data.to_numpy()
    foldIteration = 0
    foldErrors = []

#Uses K-fold to cross validate the given dataset
for trainIndices, test_indices in kFolds.split(data):

        #After each new fold, the network is reinitialized
        sess.run(initOp)

        #This starts the training phase of the MLP
        for epoch in range(amountEpochs):

            #Each epoch, result in the training set being switched
            randomTrainIndices = np.random.permutation(trainIndices)
            trainSet = data[randomTrainIndices]

            #Starts to loop the training set in order to help
            #optimize the network
            for begin in range(0, len(trainSet), batchSize):
                end = begin + batchSize
                batch_x = trainSet[begin:end].T[0].reshape((-1, 1))
                batch_y = trainSet[begin:end].T[1].reshape((-1, 1))

                sess.run(trainOp, feed_dict={
                    netInput: batch_x,
                    netTarget: batch_y,
                    dropoutProb: dropoutAmount
                })

        #This starts the testing phase of the MLP
        testSet = data[test_indices]

        #Determines the error found when completing the test set
        allError = []
        for begin in range(0, len(testSet), batchSize):
            end = begin + batchSize
            batch_x = trainSet[begin:end].T[0].reshape((-1, 1))
            batch_y = trainSet[begin:end].T[1].reshape((-1, 1))

            error = sess.run(loss, feed_dict={
                netInput: batch_x,
                netTarget: batch_y,
                dropoutProb: 1.0
            })
            allError.append(error)

        allError = np.array(allError).reshape((-1))
        foldErrors.append(allError)
        sum_mse +=np.mean(allError)

        #Displays the Error mean (MSE)and error deviation
        print("\nFold iteration:  ", foldIteration,
              "\nMSE:             ", np.mean(allError),
              "\nError deviation: ", np.std(allError),
              "\n")
        foldIteration += 1
fold_errors = np.array(foldErrors).reshape((amountFolds, -1))

hist_data = dict()
keys = ['fold 0', 'fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5', 'fold 6', 'fold 7', 'fold 8', 'fold 9']
for i, key in enumerate(keys):
    hist_data[key] = fold_errors[i]
