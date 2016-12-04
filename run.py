import matplotlib.pyplot as plt
from model import Model
import os
import tensorflow as tf

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

#Run experiment
for i in range(training):
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

    train_success_file = open(root + '/training_success.txt', 'w')
    test_success_file = open(root + '/testing_success.txt', 'w')

    if bob_results[-1] < 0.05 and abs(eve_results[-1] - N/2) > 2.0:
        #Successful run ! Now test
        train_success_file.write("%s\n" % i)
        for j in range(testing):
            success = True
            tf.reset_default_graph()
            test_model = Model(N, batch_size, learning_rate)
            test_model.restore_alice_bob(path + '/alice_bob_model.ckpt')
            eve_test_results = test_model.test(testing_epochs)
            if abs(eve_results[-1] - eve_test_results[-1]) > 1.0:
                success = False
        if success:
            #Successful testing run!
            test_success_file.write("%s\n" % i)


