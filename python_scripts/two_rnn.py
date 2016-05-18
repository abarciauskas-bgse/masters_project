import numpy as np

def loss_function(inputs, targets, hprev_l0, hprev_l1):
    assert len(inputs) == len(targets)
    T = len(inputs)
    x = {}
    h0 = {}
    h1 = {}
    # the last hidden state is given
    h0[-1] = np.copy(hprev_l0)
    h1[-1] = np.copy(hprev_l1)
    l0 = {}
    y = {}
    p_state = {}
    loss = 0
    for t in range(T):
        # structure input
        input_letter_idx = char_to_ix[inputs[t]]
        x[t] = np.zeros((vocab_size, 1))
        x[t][input_letter_idx] = 1

        # FORWARD PASS 
        # Layer 0
        h0[t] = np.tanh(np.dot(Wxh0, x[t]) + np.dot(Wh0h0, h0[(t-1)]) + bh0)
        l0[t] = np.dot(Wh0l0, h0[t]) + bl0

        # Layer 1
        # l0[t] is the input to the second layer
        h1[t] = np.tanh(np.dot(Wl0h1, l0[t]) + np.dot(Wh1h1, h1[(t-1)]) + bh1)
        y[t]  = np.dot(Wh1y, h1[t]) + by

        # Output layer
        p_state[t] = np.exp(y[t])/np.sum(np.exp(y[t]))
        target_letter_idx = char_to_ix[targets[t]]
        loss += -np.log(p_state[t][target_letter_idx, 0])
    dWxh0, dWh0h0, dbh0, dWh0l0, dbl0, dWl0h1, dWh1h1, dbh1, dWh1y, dby = np.zeros_like(Wxh0), \
        np.zeros_like(Wh0h0), \
        np.zeros_like(bh0), \
        np.zeros_like(Wh0l0), \
        np.zeros_like(bl0), \
        np.zeros_like(Wl0h1), \
        np.zeros_like(Wh1h1), \
        np.zeros_like(bh1), \
        np.zeros_like(Wh1y), \
        np.zeros_like(by)
    dh0next = np.zeros_like(h0[0])
    dh1next = np.zeros_like(h1[0])
    # Backpropagation
    for t in reversed(range(T)):
        # Backprop loss into y (e.g. loss from normalized log probabilities of y)
        dy = np.copy(p_state[t])
        target_letter_idx = char_to_ix[targets[t]]
        dy[target_letter_idx] -= 1

        # Backprop through layer 1
        dh1 = np.dot(Wh1y.T, dy) + dh1next
        dWh1y += np.dot(dy, h1[t].T)
        dby += dy
        # An update to layer 1's hidden state through non-linearity
        dh1raw = (1 - h1[t]*h1[t]) * dh1
        dbh1 += dh1raw
        dWl0h1 += np.dot(dh1raw, l0[t].T)
        dWh1h1 += np.dot(dh1raw, h1[t-1].T)
        dh1next = np.dot(Wh1h1.T, dh1raw)

        dl0 = np.dot(Wl0h1.T, dh1raw) # not sure if this should be dh1raw or dh1next
        # Backprop through layer 0 - Key difference: dy is replaced with dl0, l0 to hidden l1
        dh0 = np.dot(Wh0l0.T, dl0) + dh0next
        dWh0l0 += np.dot(dl0, h0[t].T)
        dbl0 += dl0
        dh0raw = (1 - h0[t]*h0[t]) * dh0
        dbh0 += dh0raw
        dWxh0 += np.dot(dh0raw, x[t].T)
        dWh0h0 += np.dot(dh0raw, h0[t-1].T)
        dh0next = np.dot(Wh0h0.T, dh0raw)
    for dparam in [dWxh0, dWh0h0, dbh0, dWh0l0, dbl0, dWl0h1, dWh1h1, dbh1, dWh1y, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients        
    return loss, \
        dWxh0, dWh0h0, dbh0, dWh0l0, dbl0, dWl0h1, dWh1h1, dbh1, dWh1y, dby, \
        h0[len(inputs)-1], h1[len(inputs)-1]

learning_rate = 1e-1
# number of neurons per layer (for now just one layer)
hidden_size = 10
# model parameters (x - input, l0 - layer 0 output, y - layer 1 output)
# Layer 0
Wxh0  = np.random.randn(hidden_size, vocab_size)*0.01 # input x to hidden l0
Wh0h0 = np.random.randn(hidden_size, hidden_size)*0.01 # hidden l0 to hidden l0
bh0   = np.zeros((hidden_size, 1)) # hidden bias l0
Wh0l0 = np.random.randn(vocab_size, hidden_size)*0.01 # hidden l0 to l0
bl0   = np.zeros((vocab_size, 1)) # l0 output bias
# Layer 1 - input is l0
Wl0h1 = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden l1
Wh1h1 = np.random.randn(hidden_size, hidden_size)*0.01 # hidden l1 to hidden l1
bh1   = np.zeros((hidden_size, 1)) # hidden bias l1
Wh1y  = np.random.randn(vocab_size, hidden_size)*0.01 # hidden l1 to output y
by    = np.zeros((vocab_size, 1)) # output bias
losses = []
batch_size = 25

def train(data, vocab_size, max_iters):
    # Have some memory of Weights for adagrad:
    mWxh0, mWh0h0, mbh0, mWh0l0, mbl0, mWl0h1, mWh1h1, mbh1, mWh1y, mby = \
      np.zeros_like(Wxh0), np.zeros_like(Wh0h0), np.zeros_like(bh0), np.zeros_like(Wh0l0), \
      np.zeros_like(bl0), np.zeros_like(Wl0h1), np.zeros_like(Wh1h1), np.zeros_like(bh1), np.zeros_like(Wh1y), np.zeros_like(by)
    pointer = 0
    hprev_l0 = np.zeros((hidden_size,1))
    hprev_l1 = np.zeros((hidden_size,1))
    for i in range(max_iters):
        if pointer + batch_size + 1 > len(data):
            pointer = 0
            hprev_l0 = np.zeros((hidden_size,1))
            hprev_l1 = np.zeros((hidden_size,1))
        inputs = data[pointer:(batch_size + pointer)]
        targets = data[(pointer + 1):(batch_size + pointer + 1)]
        loss, \
            dWxh0, dWh0h0, dbh0, dWh0l0, dbl0, dWl0h1, dWh1h1, dbh1, dWh1y, dby, \
            hprev_l0, hprev_l1 = loss_function(inputs, targets, hprev_l0, hprev_l1)
        # do adagrad update on parameters
        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wxh0, Wh0h0, bh0, Wh0l0, bl0, Wl0h1, Wh1h1, bh1, Wh1y, by], 
                                      [dWxh0, dWh0h0, dbh0, dWh0l0, dbl0, dWl0h1, dWh1h1, dbh1, dWh1y, dby], 
                                      [mWxh0, mWh0h0, mbh0, mWh0l0, mbl0, mWl0h1, mWh1h1, mbh1, mWh1y, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update          
        losses.append(loss)
        pointer += batch_size

    return {'losses': losses, 'hs': hprev_l0}

# def sample(h, seed_idx, nsamples):
#     sampled_idcs = []
#     x = np.zeros((vocab_size, 1))
#     x[seed_idx] = 1
#     for i in range(nsamples):
#         h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
#         y = np.dot(Why, h) + by
#         h1 = np.tanh(np.dot(Wxh1, y) + np.dot(Whh1, h) + bh1)
#         y1 = np.dot(Why1, h1) + by1
#         p = np.exp(y1)/np.sum(np.exp(y1))
#         p = np.exp(y)/np.sum(np.exp(y))
#         ix = np.random.choice(range(vocab_size), p = p.ravel())
#         x = np.zeros((vocab_size, 1))
#         x[ix] = 1
#         sampled_idcs.append(ix)
#     return sampled_idcs
