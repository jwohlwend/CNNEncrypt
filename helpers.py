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
Set of helper functions for training the CNNEncrypt adverserial networks in Tensorflow.

This project is a replication of the paper: 
"Learning to Protect Communications with Adversarial Neural Cryptography"(2016)
by Martín Abadi, David G. Andersen (Google Brain).

Authors: Jeremy Wohlwend and Luis Sanmiguel
"""

import numpy as np
import tensorflow as tf

def generate_data(batch_size, length):
	"""
	Generates plain text and key examples, where each bit is either 1.0 or -1.0

	Arguments:
	---------
		batch_size: int
			the size of the mini-batch
		length: int
			the length of each element in P and K
	Returns:
	--------
		(P, K): tuple of 2-D float32 numpy arrays
			"batch_size" examples of plaintexts P and keys K of each of size "lenght"
	"""
	P = 2 * np.random.randint(0, 2, size = (batch_size, length)) - 1
	K = 2 * np.random.randint(0, 2, size = (batch_size, length)) - 1
	return (P.astype(np.float32), K.astype(np.float32))

def weight_variable(shape, std, name):
	"""
	Generates a weight array of the given shape, initialized using a normal distribution

	Arguments:
	---------
		shape: N-D tuple
			the shape of the weight array
		std: float
			the standard deviation used in initializing the weights
		name: str
			the name to give to the variable
	Returns:
	--------
		W: tf.Variable
	"""
	initial = tf.truncated_normal(shape, stddev = std)
	W = tf.Variable(initial, name = name)
	return W

def bias_variable(shape, value, name):
	"""
	Generates a bias array of the given shape, initialized using the given value

	Arguments:
	---------
		shape: N-D tuple
			the shape of the bias array
		value: float
			the initial value of the bias variables
		name: str
			the name to give to the variable
	Returns:
	--------
		b: tf.Variable
	"""
	initial = tf.constant(value, shape = shape)
	b = tf.Variable(initial, name = name)
	return b

def fc_layer(x, shape, name):
	"""
	Implements a fully conencted layer of the given shape.

	Arguments:
	---------
		x: tensorflow object
			input variable
		shape: tuple or list
			the shape of the fully connected layer as (# inputs, # outputs)
		name: str
			the prefix for variable names
	Returns:
	--------
		tensorflow object
			the output of the fully connected layer
	"""
	outputs = shape[1]
	W = weight_variable(shape, name + "/W")
	b = bias_variable(outputs, value = 0.1, name + "/b")

	return tf.matmul(x, W) + b

def conv_layer(x, filter_shape, stride, sigmoid, name):
	"""
	Implements a 1-D convolutional layer with the given parameters.

	Arguments:
	---------
		x: tensorflow object
			input variable
		filter_shape: tuple or list
			the shape of the convolutional filter as (filter_width, # inputs, # outputs)
		stride: int
			the stride to use
		sigmoid: boolean
			if True, uses the sigmoid activation function, otherwise uses tanh
		name: str
			the prefix for variable names
	Returns:
	--------
		tensorflow object
			the output of the 1-D convolutional layer
	"""
	outputs = filter_shape[2]
	W = weight_variable(filter_shape, std = 1.0 / np.sqrt(outputs), name + "/W")
	b = bias_variable(outputs, value = 0.1, name = name + "/b")
	z = tf.nn.conv1d(x, W, stride = stride, padding = 'SAME') + b
	a = tf.sigmoid(z) if sigmoid else tf.tanh(z)
  	return a

def L1(P1, P2):
	"""
	Returns the L1 distance between plaintexts P1 and P2.

	Arguments:
	---------
		P1: tensorflow object
			the initial plaintext
		P2: tensorflow object
			the plaintext to compare
	Returns:
	--------
		tensorflow object
			the L1 distance between P1 and P2
	"""
	return tf.reduce_sum(tf.abs(tf.sub(P1, P2)), reduction_indices = 1)

def alice_bob_loss(P, Pb, N, eve_loss):
	"""
	Implements the loss function for Alice and Bob.
	The loss is computed using the L1 distance between P and Pb
	and a component weighting eve's loss such that the loss is minimum
	when Eve reconvers half the bits of P.

	Arguments:
	---------
		P: tensorflow object
			the original plaintext
		Pb: tensorflow object
			the plaintext predicted by Bob
		N: int
			the number of bits used in P
		eve_loss: tensorflow object
			eve's computed loss
	Returns:
	--------
		tensorflow object
			the loss for Alice and Bob
	"""
	return L1(P, Pb) + (((N / 2) - eve_loss)**2) / ((N / 2)**2)

def eve_loss(P, Pe):
	"""
	Implements Eve's loss function, which is simply the L1 distance between P and Pe.

	Arguments:
	---------
		P: tensorflow object
			the original plaintext
		Pe: tensorflow object
			the plaintext predicted by Eve
	Returns:
	--------
		tensorflow object
			the loss for Eve
	"""
	return L1(P, Pe)