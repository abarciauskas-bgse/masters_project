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
        input_letter_idx = char_to_ix[inputs[t]]
        x_state[t] = np.zeros((vocab_size, 1))
        # add a 1 to the x_state matrix for the current input
        x_state[t][input_letter_idx] = 1
        hidden_state[t] = np.tanh(np.dot(Wxh, x_state[t]) + np.dot(Whh, hidden_state[(t-1)]) + bh)
        y_state[t] = np.dot(Why, hidden_state[t]) + by
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
        dh = np.dot(Why.T, dy) + dhnext
        # backprop an update to hidden state through non-linearity
        dhraw = (1 - hidden_state[t]*hidden_state[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, x_state[t].T)
        dWhh += np.dot(dhraw, hidden_state[t-1].T)        
        dhnext = np.dot(Whh.T, dhraw)
    return loss, dWxh, dWhh, dWhy, dbh, dby

learning_rate = 1e-1
# number of neurons per layer (for now just one layer)
hidden_size = 10
# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh  = np.zeros((hidden_size, 1)) # hidden bias
by  = np.zeros((vocab_size, 1)) # output bias
losses = []
batch_size = 25

def train(data, vocab_size, max_iters):
    # Have some memory of Weights for adagrad:
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)
    pointer = 0
    previous_hidden_state = np.zeros((hidden_size,1))
    for i in range(max_iters):
        if pointer + batch_size + 1 > len(data):
            pointer = 0
            previous_hidden_state = np.zeros((hidden_size,1))
        inputs = data[pointer:(batch_size + pointer)]
        targets = data[(pointer + 1):(batch_size + pointer + 1)]
        loss, dWxh, dWhh, dWhy, dbh, dby = loss_function(inputs, targets, previous_hidden_state)
        # do adagrad update on parameters
        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                      [dWxh, dWhh, dWhy, dbh, dby], 
                                      [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        losses.append(loss)
        pointer += batch_size

    return {'losses': losses, 'hidden_state': previous_hidden_state}

def sample(h, seed_idx, nsamples):
    sampled_idcs = []
    x = np.zeros((vocab_size, 1))
    x[seed_idx] = 1
    for i in range(nsamples):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y)/np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p = p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        sampled_idcs.append(ix)
    return sampled_idcs





