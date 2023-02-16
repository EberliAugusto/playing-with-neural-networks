import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

np.random.seed(0)

# 8x8=64
def get_dataset(filename):
    lines = [[int(value) for value in line.split(",")]
             for line in open(filename).readlines()]

    output = [get_desired_output(line[64]) for line in lines]
    input = [np.reshape([value / 16 for value in line[0:64]], (8, 8)) for line in lines]
    value = [line[64] for line in lines]

    return np.array(output), np.array(input), np.array(value)

#returns all the network output for a given number.
def get_desired_output(number):
    desired_output =[0] * 10
    desired_output[number] = 1
    return desired_output

test_output, test_input, test_labels = get_dataset("dataset/optdigits.tes")
train_output, train_input, train_labels = get_dataset("dataset/optdigits.tra")

#model = keras.Sequential([keras.layers.Flatten(input_shape= (8,8)),
#                         keras.layers.Dense(64, activation=tf.nn.relu),
#                         keras.layers.Dense(10, activation=tf.nn.softmax)])
#model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy',  metrics=['accuracy'] )

model = keras.Sequential([keras.layers.Flatten(input_shape= (8,8)),
                         keras.layers.Dense(64, activation=tf.nn.relu),
                         keras.layers.Dense(10, activation=tf.nn.sigmoid)])
model.compile(optimizer=tf.train.GradientDescentOptimizer(0.1), loss=tf.losses.mean_squared_error,  metrics=['accuracy'])

model.fit(train_input, train_output, epochs=50)
print("--------------------------------------------------------------------------")
loss = model.evaluate(test_input, test_output)
print(loss)
print("--------------------------------------------------------------------------")
