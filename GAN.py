import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# GENERATOR
def generator(z,reuse):
    with tf.variable_scope('gen', reuse=reuse):
        hidden1 = tf.layers.dense(inputs=z,units=128)

        # Leaky ReLu activation function
        alpha = 0.01
        hidden1 = tf.maximum(alpha*hidden1,hidden1)

        hidden2 = tf.layers.dense(inputs=hidden1,units=28)

        hidden2 = tf.maximum(alph*hidden2,hidden2)

        output = tf.layers.dense(hidden2,units=784,activation=tf.nn.tanh)

        return output

# DISCRIMINATOR

def discriminator(X, reuse):
    with tf.variable_scope('dis', reuse=reuse):
        hidden1 = tf.layers.dense(inputs=X, units=128)

        # Leaky ReLu activation function
        alpha = 0.01
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        hidden2 = tf.layers.dense(inputs=hidden1, units=28)

        hidden2 = tf.maximum(alph * hidden2, hidden2)

        logits = tf.layers.dense(hidden2,units=1)
        output = tf.sigmoid(logits)

        return output

