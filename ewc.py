#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle

from functools import reduce

from pprint import pprint

# Import MNIST data
with gzip.open('mnist.pkl.gz', 'rb') as f:
    ((traind, trainl), (vald, vall), (testd, testl)
     ) = pickle.load(f, encoding='bytes')
    traind = traind.astype("float32").reshape(-1, 28, 28)
    trainl = trainl.astype("float32")
    testd = testd.astype("float32").reshape(-1, 28, 28)
    testl = testl.astype("float32")

# traind = traind.reshape(-1,784)
testd = testd.reshape(-1, 784)

# print(traind.shape)
# print(trainl.shape)

# print(testd.shape)
# print(testl.shape)


# split MNIST

# classes 0-4
classVector = trainl.argmax(axis=1)
class01234Mask = reduce(np.logical_or, [(classVector == 0), (
    classVector == 1), (classVector == 2), (classVector == 3), (classVector == 4)])

trainl_01234 = trainl[class01234Mask]
traind_01234 = traind[class01234Mask]

traind_01234 = traind_01234.reshape(-1, 784)

# print(trainl_01234.shape)
# print(traind_01234.shape)


# class 5
class5Mask = (classVector == 5)

trainl_5 = trainl[class5Mask]
traind_5 = traind[class5Mask]

traind_5 = traind_5.reshape(-1, 784)


# classes 5-9
class56789Mask = reduce(np.logical_or, [(classVector == 5), (
    classVector == 6), (classVector == 7), (classVector == 8), (classVector == 9)])

trainl_56789 = trainl[class56789Mask]
traind_56789 = traind[class56789Mask]

traind_56789 = traind_56789.reshape(-1, 784)

# print(trainl_56789.shape)
# print(traind_56789.shape)


# batch creation

def get_batch(data, labels, batch_size, index, epoch):

    # if we have exceeded the size of traind, restart!
    if index >= (data.shape[0] // batch_size):
        index = 0
        epoch += 1

    # draw batches
    dataBatch = data[index * batch_size:(index+1) * batch_size]
    labelBatch = labels[index * batch_size:(index+1) * batch_size]
    index += 1

    return dataBatch, labelBatch, index, epoch


# Parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_hidden_1 = 200  # 1st layer number of neurons
n_hidden_2 = 200  # 2st layer number of neurons
n_hidden_3 = 200  # 3st layer number of neurons
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph
X = tf.placeholder(tf.float32, [None, n_input], name='x')
Y = tf.placeholder(tf.float32, [None, n_classes], name='y')


weights = {
    'wh1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='wh1'),
    'wh2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='wh2'),
    'wh3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), name='wh3'),
    'wo': tf.Variable(tf.random_normal([n_hidden_3, n_classes]), name='wo')
}
biases = {
    'bh1': tf.Variable(tf.random_normal([n_hidden_1]), name='bh1'),
    'bh2': tf.Variable(tf.random_normal([n_hidden_2]), name='bh2'),
    'bh3': tf.Variable(tf.random_normal([n_hidden_3]), name='bh3'),
    'bo': tf.Variable(tf.random_normal([n_classes]), name='bo')
}


def neural_network(x, w, b):
    # Hidden fully connected layer with 200 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w['wh1']), b['bh1']))

    # Hidden fully connected layer with 200 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w['wh2']), b['bh2']))

    # Hidden fully connected layer with 200 neurons
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, w['wh3']), b['bh3']))

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, w['wo']) + b['bo']

    return out_layer


# Construct model
logits = neural_network(X, weights, biases)
# prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
update = optimizer.minimize(loss_op)
# print(optimizer.compute_gradients(loss_op))

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training
sess = tf.Session()
# Run the initializer
sess.run(init)


# TASK A
print("-----TASK A")
# train classes 0-4

iteration = 0
epoch = 0
for step in range(0, training_iters):
    batch_x, batch_y, iteration, epoch = get_batch(
        traind_01234, trainl_01234, batch_size, iteration, epoch)

    # Run optimization op (backprop)
    sess.run(update, feed_dict={X: batch_x, Y: batch_y})

    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={
                             X: batch_x, Y: batch_y})

        print("Step " + str(step) +
              ", Epoch" + str(epoch) +
              ", Minibatch Loss= " +
              "{:.4f}".format(loss) + ", Training Accuracy= " +
              "{:.3f}".format(acc))


# show accuracy
print("Testing Accuracy:",
      sess.run(accuracy, feed_dict={X: testd,
                                    Y: testl}))


print("TASK A")
print("Testing Accuracy:",
      sess.run(accuracy, feed_dict={X: traind_01234,
                                    Y: trainl_01234}))


# fisher calculation

zero_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0)
grads = zero_optimizer.compute_gradients(loss_op)
# pprint(grads)

gradients = {}
variables = {}

epoch_iterations = (traind_01234.shape[0] // batch_size)
epoch = 0
for step in range(0, epoch_iterations):
    batch_x, batch_y, iteration, epoch = get_batch(
        traind_01234, trainl_01234, batch_size, step, epoch)
    # print("step: " + str(step) + ", epoch: " + str(epoch))

    for grad in grads:
        grad_name = grad[1].name
        gard_name = grad_name.partition(':')[0]
        # print(gard_name)

        result = sess.run(grad, feed_dict={X: batch_x, Y: batch_y})
        gradient = result[0]

        gradient_power = np.power(gradient, 2)

        if gard_name in gradients:
            gradients[gard_name] = np.add(gradients[gard_name], gradient_power)
        else:
            gradients[gard_name] = gradient_power

        if gard_name not in variables:
            variables[gard_name] = result[1]

for key, gradient in gradients.items():
    gradients[key] = np.true_divide(gradient, epoch_iterations)


# pprint(gradients)
# for key, gradient in gradients.items():
    # print("key: ", key)
    # print("shape: ", gradient.shape)
    # print("min: ", gradient.min())
    # print("max: ", gradient.max())
    # print("---")

# print("------")




# create task A constant
taskA = {}
for key, variable in variables.items():
    taskA[key] = tf.constant(variable, name="taskA-" + key)
    # print("key: ", key)
    # print("shape: ", variable.shape)
    # print("min: ", variable.min())
    # print("max: ", variable.max())
    # print("---")


# EWC calculation

ewc_sum = 0

for key, gradient in gradients.items():

    # print("Key: ", key)

    # flatten
    varsA = tf.reshape(taskA[key], [-1])
    gradientA = tf.reshape(gradient, [-1])

    if key in weights:
        varsB = tf.reshape(weights[key], [-1])
    elif key in biases:
        varsB = tf.reshape(biases[key], [-1])

    # calc EWC appendix
    subAB = tf.subtract(varsB, varsA)
    powAB = tf.pow(subAB, 2)

    multiplyF = tf.multiply(powAB, gradientA)

    lambda_multiply = tf.multiply(multiplyF, 5)

    ewc_sum += tf.reduce_sum(lambda_multiply)




# TASK B
print("-----TASK B")

loss_op_B = tf.add(loss_op, ewc_sum)
update_B = optimizer.minimize(loss_op_B)

iteration = 0
epoch = 0
for step in range(0, training_iters):
    batch_x, batch_y, iteration, epoch = get_batch(
        traind_5, trainl_5, batch_size, iteration, epoch)

    # Run optimization op (backprop)
    sess.run(update_B, feed_dict={X: batch_x, Y: batch_y})

    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op_B, accuracy], feed_dict={
                             X: batch_x, Y: batch_y})

        print("Step " + str(step) +
              ", Epoch" + str(epoch) +
              ", Minibatch Loss= " +
              "{:.4f}".format(loss) + ", Training Accuracy= " +
              "{:.3f}".format(acc))


# show accuracy
print("Testing Accuracy:",
      sess.run(accuracy, feed_dict={X: testd,
                                    Y: testl}))


print("TASK A")
print("Testing Accuracy:",
      sess.run(accuracy, feed_dict={X: traind_01234,
                                    Y: trainl_01234}))


print("TASK B")
print("Testing Accuracy:",
      sess.run(accuracy, feed_dict={X: traind_5,
                                    Y: trainl_5}))


sess.close()
