import numpy as np

def loss_function(inputs, targets, previous_hidden_state, layer):
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
        if layer == 0:
            input_letter_idx = char_to_ix[inputs[t]]
            x_state[t] = np.zeros((vocab_size, 1))
            # add a 1 to the x_state matrix for the current input
            x_state[t][input_letter_idx] = 1
        else:
            x_state = inputs
        if layer == 0:
            hidden_state[t] = np.tanh(np.dot(Wxh, x_state[t]) + np.dot(Whh, hidden_state[(t-1)]) + bh)
            y_state[t] = np.dot(Why, hidden_state[t]) + by
        else:
            hidden_state[t] = np.tanh(np.dot(Wxh1, x_state[t]) + np.dot(Whh1, hidden_state[(t-1)]) + bh1)
            y_state[t] = np.dot(Why1, hidden_state[t]) + by1
        p_state[t] = np.exp(y_state[t])/np.sum(np.exp(y_state[t]))
        target_letter_idx = char_to_ix[targets[t]]
        loss += -np.log(p_state[t][target_letter_idx,0])
    dWxh, dWhh, dWhy, dbh, dby = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why), np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hidden_state[0])
    for t in reversed(range(T)):
        dy = np.copy(p_state[t])
        # get probability of correct the target from forward pass and subtract 1 to backprop a loss into y
        target_letter_idx = char_to_ix[targets[t]]
        dy[target_letter_idx] -= 1
        dWhy += np.dot(dy, hidden_state[t].T)
        dby += dy
        if layer == 0:
            dh = np.dot(Why.T, dy) + dhnext
        else:
            dh = np.dot(Why1.T, dy) + dhnext
        # backprop an update to hidden state through non-linearity
        dhraw = (1 - hidden_state[t]*hidden_state[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, x_state[t].T)
        dWhh += np.dot(dhraw, hidden_state[t-1].T)
        if layer == 0:
            dhnext = np.dot(Whh.T, dhraw)
        else:
            dhnext = np.dot(Whh1.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients        
    return loss, dWxh, dWhh, dWhy, dbh, dby, hidden_state[len(inputs)-1], y_state

learning_rate = 1e-1
# number of neurons per layer (for now just one layer)
hidden_size = 10
# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh  = np.zeros((hidden_size, 1)) # hidden bias
by  = np.zeros((vocab_size, 1)) # output bias
Wxh1 = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh1 = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why1 = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh1  = np.zeros((hidden_size, 1)) # hidden bias
by1  = np.zeros((vocab_size, 1)) # output bias
losses = []
batch_size = 25

def train(data, vocab_size, max_iters):
    # Have some memory of Weights for adagrad:
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mWxh1, mWhh1, mWhy1 = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)
    mbh1, mby1 = np.zeros_like(bh), np.zeros_like(by)
    pointer = 0
    previous_hidden_state = np.zeros((hidden_size,1))
    previous_hidden_state1 = np.zeros((hidden_size,1))
    for i in range(max_iters):
        if pointer + batch_size + 1 > len(data):
            pointer = 0
            previous_hidden_state = np.zeros((hidden_size,1))
            previous_hidden_state1 = np.zeros((hidden_size,1))
        inputs = data[pointer:(batch_size + pointer)]
        targets = data[(pointer + 1):(batch_size + pointer + 1)]
        loss, dWxh, dWhh, dWhy, dbh, dby, previous_hidden_state, y_state = loss_function(inputs, targets, previous_hidden_state, 0)
        loss1, dWxh1, dWhh1, dWhy1, dbh1, dby1, previous_hidden_state1, _ = loss_function(y_state, targets, previous_hidden_state1, 1)
        # do adagrad update on parameters
        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                      [dWxh, dWhh, dWhy, dbh, dby], 
                                      [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        for param, dparam, mem in zip([Wxh1, Whh1, Why1, bh1, by1], 
                                      [dWxh1, dWhh1, dWhy1, dbh1, dby1], 
                                      [mWxh1, mWhh1, mWhy1, mbh1, mby1]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update            
        losses.append(loss1)
        pointer += batch_size

    return {'losses': losses, 'hidden_state': previous_hidden_state}

def sample(h, seed_idx, nsamples):
    sampled_idcs = []
    x = np.zeros((vocab_size, 1))
    x[seed_idx] = 1
    for i in range(nsamples):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        h1 = np.tanh(np.dot(Wxh1, y) + np.dot(Whh1, h) + bh1)
        y1 = np.dot(Why1, h1) + by1
        p = np.exp(y1)/np.sum(np.exp(y1))
        p = np.exp(y)/np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p = p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        sampled_idcs.append(ix)
    return sampled_idcs
