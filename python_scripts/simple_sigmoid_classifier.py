# Sigmoid classifier via gradient descent
# Single neuron as a linear classifier
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Simulate X and Y
# XOR pattern
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([[0,1,1,0]]).T

# Initial weights, a set for class 0 and a set for class 1
# W is dimension (dim of X x number of classes); an array of weights
# python does it cols x rows
# 1 - like a nnet with 1 neuron
W = 2*np.random.random((3,4)) - 1

# Sigmoid activation function gives us a probability for classes (1,0)
i = 3
obs = X[i,:]
y_obs = Y[i]
losses = []
iters = 1000

# first neuron's weights
weights_neuron1 = W[:,i]

for iter in range(iters):
    print 'starting iter: ' + str(iter)
    # calculate the input to the first neuron
    input_neuron1 = np.dot(weights_neuron1, obs) 
    # calculate activation function
    # sigmoid in this case
    f_neuron1 = 1/(1+np.exp(-input_neuron1))
    # this is the ouptput of the neuron
    # if y = 1, f_neuron is the correct probability
    # if y = 0, 1-f_neuron1 is the correct probability
    true_class_prob = f_neuron1 if y_obs == 1 else (1-f_neuron1)
    loss_neuron1 = -true_class_prob + np.log(np.sum([np.exp(f_neuron1), np.exp(1-f_neuron1)]))
    #loss_neuron1 = 1 - true_class_prob
    # equivalent to evaluating: 
    # loss_neuron1_simple = np.abs(y_obs - f_neuron1)
    # print 'loss: ' + str(loss_neuron1 - loss_neuron1_simple)
    losses.append(loss_neuron1)
    dLdF = 1 if y_obs == 1 else -1
    dL = loss_neuron1 * dLdF
    # the gradient for this neuron and this input
    dW = dL * f_neuron1 * (1 - f_neuron1)
    #weights_neuron1 += np.dot(obs, dW)
    weights_neuron1 += obs * dW


x = range(iters)
line = plt.plot(x, np.array(losses), linewidth=2)
plt.show()
