# Provided under the MIT License (MIT)
# Copyright (c) 2016 Jeremy Wohlwend, Luis Sanmiguel

# Permission is hereby granted, free of charge, to any person obtaining 
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Main file containing the CNNEncrypt adverserial neural network model,
and methods for training, testing and saving the model.

This project is a replication of the paper: 
"Learning to Protect Communications with Adversarial Neural Cryptography"(2016)
by Martin Abadi, David G. Andersen (Google Brain).

Authors: Jeremy Wohlwend and Luis Sanmiguel
"""

import tensorflow as tf
from helpers import *

class Model:
    """
    Implements the Adverserial Neural Network model.

    The model contains 3 neural networks: Alice, Bob and Eve. 
    Alice takes in a plaintext P and a key K and produces a cypher C.
    Bob attempts to reconstruct P from C and K, while Eve tries to decypher C
    without any knowledge of the key K.

    Each neural network is composed of a fully connected layer and 4 convolution layers.
    These all use sigmoid activation functions except for the output layer which uses tanh.

    More detail about the loss functions and the format of the layers can be found in 
    the helpers.py file.
    """
    def __init__(self, N, batch_size, learning_rate):
        """
        Initializes a model with the given parameters.

        Arguments:
        ---------
            N: int
                the number of bits in the plaintext P
                the same number is used for the size of the key K
            batch_size: int
                the mini-batch size to use in training
            learning_rate: float
                the constant learning rate used by the Adam optimizer
        """

        #Create session
        self.sess = tf.InteractiveSession()

        #Below P refers to plaintext input, K to symmetric key and C to cypher text 
        #Pb is Bob's prediction of the plaintext given C and K
        #Pe is Eve's prediction of the plaintext given C

        #Input size of 2 * N: len(K) + len(P)
        #Output size of N: len(C)
        self.P, self.K = generate_data(batch_size, N)

        #Setting sigmoid to False changes to activation function to tanh

        #Alice's network
        alice_input = tf.concat(1, [self.P, self.K])
        alice_fc = fc_layer(alice_input, shape = (2 * N, 2 * N), name = 'alice_bob/alice_fc')
        alice_fc = tf.reshape(alice_fc, [batch_size, 2 * N, 1])
        alice_conv1 = conv_layer(alice_fc, filter_shape = [4, 1, 2], stride = 1, sigmoid = True, name = 'alice_bob/alice_conv1')
        alice_conv2 = conv_layer(alice_conv1, filter_shape = [2, 2, 4], stride = 2, sigmoid = True, name = 'alice_bob/alice_conv2')
        alice_conv3 = conv_layer(alice_conv2, filter_shape = [1, 4, 4], stride = 1, sigmoid = True, name = 'alice_bob/alice_conv3')
        alice_conv4 = conv_layer(alice_conv3, filter_shape = [1, 4, 1], stride = 1, sigmoid = False, name = 'alice_bob/alice_conv4')
        self.C = tf.reshape(alice_conv4, [batch_size, N])

        #Bob's network
        bob_input = tf.concat(1, [self.C, self.K])
        bob_fc = fc_layer(bob_input, shape = (2 * N, 2 * N), name = 'alice_bob/bob_fc')
        bob_fc = tf.reshape(bob_fc, [batch_size, 2 * N, 1])
        bob_conv1 = conv_layer(bob_fc, filter_shape = [4, 1, 2], stride = 1, sigmoid = True, name = 'alice_bob/bob_conv1')
        bob_conv2 = conv_layer(bob_conv1, filter_shape = [2, 2, 4], stride = 2, sigmoid = True, name = 'alice_bob/bob_conv2')
        bob_conv3 = conv_layer(bob_conv2, filter_shape = [1, 4, 4], stride = 1, sigmoid = True, name = 'alice_bob/bob_conv3')
        bob_conv4 = conv_layer(bob_conv3, filter_shape = [1, 4, 1], stride = 1, sigmoid = False, name = 'alice_bob/bob_conv4')
        Pb = tf.reshape(bob_conv4, [batch_size, N])

        #Eve's network
        eve_fc = fc_layer(self.C, shape = (N, 2 * N), name = 'eve/fc')
        eve_fc = tf.reshape(eve_fc, [batch_size, 2 * N, 1])
        eve_conv1 = conv_layer(eve_fc, filter_shape = [4, 1, 2], stride = 1, sigmoid = True, name = 'eve/conv1')
        eve_conv2 = conv_layer(eve_conv1, filter_shape = [2, 2, 4], stride = 2, sigmoid = True, name = 'eve/conv2')
        eve_conv3 = conv_layer(eve_conv2, filter_shape = [1, 4, 4], stride = 1, sigmoid = True, name = 'eve/conv3')
        eve_conv4 = conv_layer(eve_conv3, filter_shape = [1, 4, 1], stride = 1, sigmoid = False, name = 'eve/conv4')
        Pe = tf.reshape(eve_conv4, [batch_size, N])

        #Compute loss
        eve_loss = eve_loss_function(P, Pe)
        alice_bob_loss = alice_bob_loss_function(P, Pb, Pe, N)

        #Compute number of bits wrong for Bob and Eve
        self.bob_bit_error = get_bit_error(P, Pb)
        self.eve_bit_error = get_bit_error(P, Pe)

        #Define optimizer and learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate)

        #Get all variables
        self.alice_bob_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "alice_bob/")
        self.eve_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "eve/")

        #Define training step
        self.alice_bob_train_step = optimizer.minimize(alice_bob_loss, var_list=self.alice_bob_vars)
        self.eve_train_step = optimizer.minimize(eve_loss, var_list=self.eve_vars)

        #Prepare saver
        self.alice_bob_saver = tf.train.Saver(self.alice_bob_vars)
        self.eve_saver = tf.train.Saver(self.eve_vars)

        #Initialize
        self.sess.run(tf.initialize_all_variables())

    def train(self, epochs):
        """
        Trains the model by running Alice and Bob for one run and Eve for two.
        This gives Eve a slight computational edge.

        Arguments:
        ---------
            epochs: int
                the number of training iterations
        
        Returns:
        --------
            bob_results: list of floats
                bob's mean bit error for every 100th iteration
            eve_results: list of floats
                eve's mean bit error for every 100th iteration
        """
        bob_results = []
        eve_results = []
        for i in xrange(epochs):
            if i % 100 == 0:
                training_error = self.bob_bit_error.eval(), self.eve_bit_error.eval()
                print("step {}, bit error {}".format(i, training_error))
                bob_results.append(training_error[0])
                eve_results.append(training_error[1])
            #Train Alice and Bob
            self.alice_bob_train_step.run()
            #Train Eve twice
            self.eve_train_step.run()
            self.eve_train_step.run()

        return bob_results, eve_results

    def test(self, epochs):
        """
        Tests the model by running Eve alone for epochs iterations

        Arguments:
        ---------
            epochs: int
                the number of testing iterations
        
        Returns:
        --------
            eve_results: list of floats
                eve's mean bit error for every 1000th iteration
        """
        eve_results = []
        for i in xrange(epochs):
            if i % 1000 == 0:
                training_error = self.eve_bit_error.eval()
                print("step {}, bit error {}".format(i, training_error))
                eve_results.append(training_error)
            self.eve_train_step.run()

        return eve_results

    def analyse(self):
        """
        Analyses the model by evaluating a single batch

        Returns
        -------
            P: array of shape [batch_size, N]
                the original plaintext
            K: array of shape [batch_size, N]
                the corresponding key
            C: array of shape [batch_size, N]
                the cypher generated by Alice
        """
        return(self.P.eval(), self.K.eval(), self.C.eval())

    def restore_alice_bob(self, restore_path):
        """
        Restores eve from the given file

        Arguments:
        ---------
            restore_path: string
                the path to alice and bob's data
        """
        self.alice_bob_saver.restore(self.sess, restore_path)

    def restore_eve(self, restore_path):
        """
        Restores eve from the given file

        Arguments:
        ---------
            restore_path: string
                the path to eve's data
        """
        self.eve_saver.restore(self.sess, restore_path)

    def save(self, save_path):
        """
        Saves the model to the given location

        Arguments:
        ---------
            save_path: string
                the directory to use to store the model
        """
        self.alice_bob_saver.save(self.sess, save_path + "/alice_bob_model.ckpt")
        self.eve_saver.save(self.sess, save_path + "/eve_model.ckpt")



