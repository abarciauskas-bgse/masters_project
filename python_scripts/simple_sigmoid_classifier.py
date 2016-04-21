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
loss_all = []

# Sigmoid activation function
for i in range(len(X)):
    obs = X[i,:]
    y_obs = Y[i]
    losses = []
    iters = 1000
    # first neuron's weights
    # we have four sets for future times where we have more than one neuron
    weights_neuron1 = W[:,0]
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
        loss_neuron1 = 1 - true_class_prob
        losses.append(loss_neuron1)
        dLdF = 1 if y_obs == 1 else -1
        dL = loss_neuron1 * dLdF
        # the gradient for this neuron and this input
        dW = dL * f_neuron1 * (1 - f_neuron1)
        #weights_neuron1 += np.dot(obs, dW)
        weights_neuron1 += obs * dW
    loss_all.append(losses)


x = range(iters)
line = plt.plot(x, loss_all[0],  x, loss_all[1], x, loss_all[2], x, loss_all[3], linewidth=2)
plt.show()
