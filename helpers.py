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
by Martin Abadi, David G. Andersen (Google Brain).

Authors: Jeremy Wohlwend and Luis Sanmiguel
"""
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
	P = 2 * tf.random_uniform([batch_size, length], minval=0, maxval=2, dtype=tf.int32) - 1
	K = 2 * tf.random_uniform([batch_size, length], minval=0, maxval=2, dtype=tf.int32) - 1
	return (tf.to_float(P), tf.to_float(K))

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
	num_inputs, num_outputs = shape
	W = weight_variable(shape, 1.0, name + "/W")
	b = bias_variable([num_outputs], 0.0, name + "/b")
	return tf.nn.sigmoid(tf.matmul(x, W) + b)


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
	filter_width, num_inputs, num_outputs = filter_shape
	W = weight_variable(filter_shape, 0.1, name + "/W")
	b = bias_variable([num_outputs], 0.0, name + "/b")
	z = tf.nn.conv1d(x, W, stride = stride, padding = 'SAME') + b
	a = tf.nn.sigmoid(z) if sigmoid else tf.nn.tanh(z)
  	return a

def L1(P1, P2):
	"""
	Returns the L1 distance between plaintexts P1 and P2,
	averaged over the number of batches

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
	#Suggested by Professor Andersen
	bits_wrong = tf.reduce_sum(tf.abs((P1 + 1.0) / 2.0 - (P2 + 1.0) / 2.0), [1])
	return tf.reduce_mean(bits_wrong)

def alice_bob_loss_function(P, Pb, N, eve_loss):
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
	return L1(P, Pb) + ((N / 2 - eve_loss)**2) / (N / 2)**2

def eve_loss_function(P, Pe):
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

def get_bit_error(P1, P2):
	"""
	Returns the number of bits that are different between P1 and P2
	P1 and P2 are first mapped to bit values using the sign function and then compared

	Arguments:
	---------
		P1: tensorflow object
			first plaintext as float or bit values
		P2: tensorflow object
			second plaintext as float or bit values
	Returns:
	--------
		tensorflow object
			the number of different bits between P1 and P2
	"""
	boolean_error = tf.cast(tf.not_equal(tf.sign(P1), tf.sign(P2)), tf.float32)
	return tf.reduce_mean(tf.reduce_sum(boolean_error, [1]))

