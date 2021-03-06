import numpy as np

def loss_function(inputs, targets, previous_hidden_state):
    assert len(inputs) == len(targets)
    T = len(inputs)
    x_state = {}
    hidden_state = {}
    # the last hidden state is given
    hidden_state[-1] = np.copy(previous_hidden_state)
    y_state = {}
    p_state = {}
    loss = 0
    for t in range(T):
        x_state[t] = np.array([inputs[t]]).T
        hidden_state[t] = np.tanh(np.dot(Wxh, x_state[t]) + np.dot(Whh, hidden_state[(t-1)]) + bh)
        y_state[t] = np.dot(Why, hidden_state[t]) + by
        p_state[t] = np.exp(y_state[t])/np.sum(np.exp(y_state[t]))
        interim_loss = 0
        # FIXME: hack for loss
        for idx, target_note in enumerate(targets[t]):
            if target_note == 1: loss += -np.log(p_state[t][idx,0])
            break
    dWxh, dWhh, dWhy, dbh, dby = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why), np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hidden_state[0])
    for t in reversed(range(T)):
        dy = np.copy(p_state[t])
        # get probability of correct the target from forward pass and subtract 1 to backprop a loss into y
        for idx, target_note in enumerate(targets[t]):
            if target_note == 1: dy[idx] -= 1
        dWhy += np.dot(dy, hidden_state[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        # backprop an update to hidden state through non-linearity
        dhraw = (1 - hidden_state[t]*hidden_state[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, x_state[t].T)
        dWhh += np.dot(dhraw, hidden_state[t-1].T)        
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients        
    return loss, dWxh, dWhh, dWhy, dbh, dby, hidden_state[len(inputs)-1]

learning_rate = 1e-1
# number of neurons per layer (for now just one layer)
hidden_size = 200
# model parameters
Wxh = np.random.randn(hidden_size, num_classes)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(num_classes, hidden_size)*0.01 # hidden to output
bh  = np.zeros((hidden_size, 1)) # hidden bias
by  = np.zeros((num_classes, 1)) # output bias
losses = []

def train(data, num_classes, max_iters, batch_size = 25):
    # Have some memory of Weights for adagrad:
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)
    pointer = 0
    previous_hidden_state = np.zeros((hidden_size,1))
    # loss at iter 0
    smooth_loss = -np.log(1.0/num_classes)*batch_size
    for i in range(max_iters):
        if pointer + batch_size + 1 > len(data):
            pointer = 0
            previous_hidden_state = np.zeros((hidden_size,1))
        inputs = data[pointer:(batch_size + pointer)]
        targets = data[(pointer + 1):(batch_size + pointer + 1)]
        loss, dWxh, dWhh, dWhy, dbh, dby, previous_hidden_state = loss_function(inputs, targets, previous_hidden_state)
        # do adagrad update on parameters
        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                      [dWxh, dWhh, dWhy, dbh, dby], 
                                      [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        losses.append(smooth_loss)
        pointer += batch_size

    return {'losses': losses, 'hidden_state': previous_hidden_state}

def sample(h, seed_idx, nsamples):
    sampled_idcs = []
    x = np.zeros((num_classes, 1))
    x[seed_idx] = 1
    for i in range(nsamples):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y)/np.sum(np.exp(y))
        ix = np.random.choice(range(num_classes), p = p.ravel())
        x = np.zeros((num_classes, 1))
        x[ix] = 1
        sampled_idcs.append(ix)
    return sampled_idcs

