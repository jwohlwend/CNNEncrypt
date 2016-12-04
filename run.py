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
Script to run experiments using the CNNEncrypt adverserial networks

This project is a replication of the paper: 
"Learning to Protect Communications with Adversarial Neural Cryptography"(2016)
by Martin Abadi, David G. Andersen (Google Brain).

Authors: Jeremy Wohlwend and Luis Sanmiguel
"""
import matplotlib.pyplot as plt
from model import Model
import tensorflow as tf
import os

#Experiment parameters
training = 20
training_epochs = 20000
testing = 5
testing_epochs = 100000

#Model parameters
N = 16
batch_size = 4096
learning_rate = 0.0008

#Save folder
root = "/home/jeremy/Documents/experiment1"
train_success_file = open(root + '/training_success.txt', 'a')
test_success_file = open(root + '/testing_success.txt', 'a')

#Run experiment
for i in range(6, training):
    #Init
    tf.reset_default_graph()
    path = root + '/model_' + str(i)
    if not os.path.exists(path):
        os.makedirs(path)
    model = Model(N, batch_size, learning_rate)
    #Train
    bob_results, eve_results = model.train(training_epochs)
    #Save
    plt.figure()
    plt.plot(range(0, training_epochs, 100), bob_results, range(0, training_epochs, 100), eve_results)
    plt.xlabel('training iteration', fontsize=16)
    plt.ylabel('bit error', fontsize=16)
    plt.savefig(path + "/fig.png")
    model.save(path)

    if bob_results[-1] < 0.05 and abs(eve_results[-1] - N/2) < 2.0:
        #Successful run ! Now test
        train_success_file.write("%s\n" % i)
        success = True
        for j in range(testing):
            tf.reset_default_graph()
            test_model = Model(N, batch_size, learning_rate)
            test_model.restore_alice_bob(path + '/alice_bob_model.ckpt')
            eve_test_results = test_model.test(testing_epochs)
            if abs(eve_results[-1] - eve_test_results[-1]) > 1.0:
                success = False
                break
        if success:
            #Successful testing run!
            test_success_file.write("%s\n" % i)

train_success_file.close()
test_success_file.close()

