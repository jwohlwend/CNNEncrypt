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
Simple script to train the CNNEncrypt adverserial neural networks.

This project is a replication of the paper: 
"Learning to Protect Communications with Adversarial Neural Cryptography"(2016)
by Martín Abadi, David G. Andersen (Google Brain).

Authors: Jeremy Wohlwend and Luis Sanmiguel
"""

import tensorflow as tf
from helpers import *

#Below P refers to plaintext input, K to symmetric key and C to cypher text
#Pb is Bob's prediction of the plaintext given C and K
#Pe is Eve's prediction of the plaintext given C

#Choose N, the size of the pliantext in bits. 
#The same value is used for the size of the symmetric key K
#Batch_size is the number of examples to use in a single iterations
#Epochs is the number of training iterations to run
N = 16
batch_size = 512
epochs = 1000

#Create session
sess = tf.InteractiveSession()

#Input size of 2 * N: len(K) + len(P)
#Output size of N: len(C)
P = tf.placeholder(tf.float32, shape = [batch_size, N])
K = tf.placeholder(tf.float32, shape = [batch_size, N])

#Setting sigmoid to False changes to activation function to tanh

#Alice's network
alice_input = tf.concat(1, [P, K])
alice_fc 	= fc_layer(alice_input, layer_shape = (2 * N, 2 * N), name = 'alice_bob/alice_fc')
alice_conv1 = conv_layer(alice_fc, filter_shape = [4, 1, 2], stride = 1, sigmoid = True, name = 'alice_bob/alice_conv1')
alice_conv2 = conv_layer(alice_conv1, filter_shape = [2, 2, 4], stride = 2, sigmoid = True, name = 'alice_bob/alice_conv2')
alice_conv3 = conv_layer(alice_conv2, filter_shape = [1, 4, 4], stride = 1, sigmoid = True, name = 'alice_bob/alice_conv3')
C 			= conv_layer(alice_conv3, filter_shape = [1, 4, 1], stride = 1, sigmoid = False, name = 'alice_bob/alice_conv4')

#Bob's network
bob_input 	= tf.concat(1, [C, K])
bob_fc 		= fc_layer(bob_input, layer_shape = (2 * N, 2 * N), name = 'alice_bob/bob_fc')
bob_conv1 	= conv_layer(bob_fc, filter_shape = [4, 1, 2], stride = 1, sigmoid = True, name = 'alice_bob/bob_conv1')
bob_conv2 	= conv_layer(bob_conv1, filter_shape = [2, 2, 4], stride = 2, sigmoid = True, name = 'alice_bob/bob_conv2')
bob_conv3 	= conv_layer(bob_conv2, filter_shape = [1, 4, 4], stride = 1, sigmoid = True, name = 'alice_bob/bob_conv3')
Pb 			= conv_layer(bob_conv3, filter_shape = [1, 4, 1], stride = 1, sigmoid = False, name = 'alice_bob/bob_conv4')

#Eve's network
eve_fc 		= fc_layer(C, layer_shape = (N, 2 * N), name = 'eve/fc')
eve_conv1 	= conv_layer(eve_fc, filter_shape = [4, 1, 2], stride = 1, sigmoid = True, name = 'eve/conv1')
eve_conv2 	= conv_layer(eve_conv1, filter_shape = [2, 2, 4], stride = 2, sigmoid = True, name = 'eve/conv2')
eve_conv3 	= conv_layer(eve_conv2, filter_shape = [1, 4, 4], stride = 1, sigmoid = True, name = 'eve/conv3')
Pe 			= conv_layer(eve_conv3, filter_shape = [1, 4, 1], stride = 1, sigmoid = False, name = 'eve/conv4')

#Compute loss
eve_loss = eve_loss_function(P, Pe)
alice_bob_loss = alice_bob_loss_function(P, Pb, N, loss_e)

#Compute mean error
eve_error = tf.reduce_mean(loss_e)
alice_bob_error = tf.reduce_mean(loss_ab)

#Define optimizer and learning rate
optimizer = tf.train.AdamOptimizer(0.0008)

#Get all variables
alice_bob_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "alice_bob/")
eve_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "eve/")

#Define training step
alice_bob_train_step = optimizer.minimize(loss_ab, var_list=alice_bob_vars)
eve_train_step = optimizer.minimize(loss_e, var_list=eve_vars)

#Train
sess.run(tf.initialize_all_variables())
for i in xrange(epochs):
  	if i % 100 == 0:
    	training_error = alice_bob_error.eval(), eve_error.eval()
    	print("step {}, training error {}".format(i, training_error))
    #Train Alice and Bob once
  	alice_bob_train_step.run()
  	#Train Eve twice
  	eve_train_step.run()
  	eve_train_step.run()