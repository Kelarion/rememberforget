"""
Classes used in the Remember-Forget experiments.
 
Includes:
    - RNNModel: class that currently supports {'LSTM', 'GRU', 'tanh', 'relu','tanh-GRU', 'relu-GRU'}
    recurrent neural networks. includes a save, load, and train method.
    - stateDecoder: does the memory decoding. includes a train method.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from enum import IntEnum
import numpy as np
import scipy.linalg as la
import scipy.special as spc

#%%
class RNNModel(nn.Module):
    """ 
    Skeleton for a couple RNNs 
    Namely: rnn_type = {'LSTM', 'GRU', 'tanh', 'relu', 'tanh-GRU', 'relu-GRU'}
    The final two (tanh-GRU and relu-GRU) use the custom GRU class.
    """
    
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, 
        embed=False, persistent=False, padding=-1):
        super(RNNModel,self).__init__()

        if embed:
            self.encoder = nn.Embedding(ntoken, ninp, padding_idx=padding)
        else:
            if persistent:
                self.encoder = ContextIndicator(ntoken, ninp, padding_idx=padding) # defined below
            else:
                self.encoder = Indicator(ntoken, ninp, padding_idx=padding) # defined below
            
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        elif rnn_type in ['tanh-GRU', 'relu-GRU']:
            nlin = getattr(torch, rnn_type.split('-GRU')[0])
            self.rnn = CustomGRU(ninp, nhid, nlayers, nonlinearity=nlin) # defined below
        else:
            try:
                self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=rnn_type)
            except:
                raise ValueError("Invalid rnn_type: give from {'LSTM', 'GRU', 'tanh', 'relu'}")

        self.decoder = nn.Linear(nhid, ntoken)
        self.softmax = nn.LogSoftmax(dim=2)
        
        self.embed = embed
        self.init_weights()
        
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.padding = padding

    def init_weights(self):
        if self.embed:
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_()

    def forward(self, input, hidden, give_gates=False, debug=False):
        """
        Run the RNN forward. Expects input to be (lseq,nseq,...)
        Only set give_gates=True if it's the custom GRU!!!
        use `debug` argument to return also the embedding the input
        """

        emb = self.encoder(input)
        if emb.dim()<3:
            emb = emb.unsqueeze(0)

        if give_gates:
            output, hidden, extras = self.rnn(emb, hidden, give_gates)
        else:
            output, hidden = self.rnn(emb, hidden)

        decoded = self.softmax(self.decoder(output))
        # decoded = self.decoder(output)

        if give_gates:
            if debug:
                return decoded, hidden, extras, emb
            else:
                return decoded, hidden, extras
        else:
            if debug:
                return decoded, hidden, emb
            else:
                return decoded, hidden

    def test_inputs(self, seqs, padding=-1):
        """ 
        for debugging purposes: convert seqs to appropriate type and
        shape to give into rnn.forward(), and also run init_hidden

        seqs should be (nseq, lseq), i.e. output of make_dset or draw_seqs
        """
        inp = torch.tensor(seqs.T).type(torch.LongTensor)
        inp[inp==padding] = 0

        hid = self.init_hidden(seqs.shape[0])

        return inp, hid

    def init_hidden(self, bsz):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(1, bsz, self.nhid),
                    torch.zeros(1, bsz, self.nhid))
        else:
            return torch.zeros(1, bsz, self.nhid)

    def save(self, to_path):
        """
        save model parameters to path
        """
        with open(to_path, 'wb') as f:
            torch.save(self.state_dict(), f)
    
    def load(self, from_path):
        """
        load parameters into model
        """
        with open(from_path, 'rb') as f:
            self.load_state_dict(torch.load(f))

    ### specific for our task (??)
    def train(self, X, Y, optparams, dlparams, algo=optim.SGD,
        criterion=nn.NLLLoss(), nepoch=1000, do_print=True,
        epsilon=0, padding=-1):
        """
        Train rnn on data X and labels Y (both torch tensors).
        X, Y need to have samples as the FIRST dimension
        """
        
        self.optimizer = algo(self.parameters(), **optparams)
        padding = self.padding

        dset = torch.utils.data.TensorDataset(X, Y)
        trainloader = torch.utils.data.DataLoader(dset, **dlparams)
        
        loss_ = np.zeros(0)
        prev_loss = 0
        for epoch in range(nepoch):
            running_loss = 0.0
            
            for i, batch in enumerate(trainloader, 0):
                # unpack
                btch, labs = batch
                btch = btch.transpose(1,0) # (lenseq, nseq, ninp)
                
                # initialise
                self.optimizer.zero_grad()
                hidden = self.init_hidden(btch.size(1))
                
                # forward -> backward -> optimize
                output = torch.zeros(btch.size(1), 
                                     self.decoder.out_features)

                # we're going to train on the final non-padding time point
                t_final = -(np.flipud(btch!=self.padding).argmax(0)+1) # index of final time point
                # btch[btch==self.padding] = 0 # set to some arbitrary value

                out, hidden = self(btch, hidden)
                output = out[t_final, np.arange(btch.size(1)), :]

                loss = criterion(output.squeeze(0),labs.long().squeeze()) # | || || |_
                loss.backward()

                self.optimizer.step()

                # update loss
                running_loss += loss.item()
                loss_ = np.append(loss_, loss.item())
                
            if (epsilon>0) and (np.abs(running_loss-prev_loss) <= epsilon):
                print('~'*5)
                print('[%d] Converged at loss = %0.3f'%(epoch+1, running_loss/(i+1)))
                return loss_
            # print to screen
            if do_print:
                print('[%d] loss: %.3f' %
                      (epoch + 1, running_loss / i))
                running_loss = 0.0
                prev_loss = running_loss

        print('[%d] Finished at loss = %0.3f'%(epoch+1, running_loss/(i+1)))
        print('~'*5)
        return loss_

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Custom GRU, originally by Miguel but substantially changed
class CustomGRU(nn.Module):
    """
    A GRU class which gives access to the gate activations during a forward pass

    Supposed to mimic the organisation of torch.nn.GRU -- same parameter names
    """
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity=torch.tanh):
        """Mimics the nn.GRU module. Currently num_layers is not supported"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input weights
        self.weight_ih_l0 = Parameter(torch.Tensor(3*hidden_size, input_size))

        # hidden weights
        self.weight_hh_l0 = Parameter(torch.Tensor(3*hidden_size, hidden_size))

        # bias
        self.bias_ih_l0 = Parameter(torch.Tensor(3*hidden_size)) # input
        self.bias_hh_l0 = Parameter(torch.Tensor(3*hidden_size)) # hidden

        self.f = nonlinearity

        self.init_weights()

    def __repr__(self):
        return "CustomGRU(%d,%d)"%(self.input_size,self.hidden_size)

    def init_weights(self):
        for p in self.parameters():
            k = np.sqrt(self.hidden_size)
            nn.init.uniform_(p.data, -k, k)

    def forward(self, x, init_state, give_gates=False):
        """Assumes x is of shape (len_seq, batch, input_size)"""
        seq_sz, bs, _ = x.size()

        update_gates = torch.empty(seq_sz, bs, self.hidden_size)
        reset_gates = torch.empty(seq_sz, bs, self.hidden_size)
        hidden_states = torch.empty(seq_sz, bs, self.hidden_size)

        h_t = init_state

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[t, :, :]

            gi = F.linear(x_t, self.weight_ih_l0, self.bias_ih_l0) # do the matmul all together
            gh = F.linear(h_t, self.weight_hh_l0, self.bias_hh_l0)

            i_r, i_z, i_n = gi.chunk(3,1) # input currents
            h_r, h_z, h_n = gh.chunk(3,2) # hidden currents

            r_t = torch.sigmoid(i_r + h_r)
            z_t = torch.sigmoid(i_z + h_z)
            n = self.f(i_n + r_t*h_n)
            h_t = n + z_t*(h_t - n)

            update_gates[t,:,:] = z_t
            reset_gates[t,:,:] = r_t
            hidden_states[t,:,:] = h_t

        output = hidden_states

        if give_gates:
            return output, h_t, (update_gates, reset_gates)
        else:
            return output, h_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% encoders 
class Indicator(nn.Module):
    """
    class to implement indicator function (i.e. one-hot encoding)
    it's a particular type of embedding, which isn't trainable

    by default, `-1` maps to all zeros. change this by setting the `padding` argument
    """
    def __init__(self, ntoken, ninp, padding_idx=-1):
        super().__init__()
        self.ntoken = ntoken
        self.ninp = ntoken

        self.padding = padding_idx

    def __repr__(self):
        return "Indicator(ntoken=%d, ninp=%d)"%(self.ntoken,self.ntoken)

    def forward(self, x):
        """
        Convert list of sequences x to an indicator (i.e. one-hot) representation
        extra dimensions are added at the end
        """

        def all_idx(idx, axis):
            """ from stack exchange"""
            grid = np.ogrid[tuple(map(slice, idx.shape))]
            grid.insert(axis, idx)
            return tuple(grid)
        
        ignore = np.repeat(np.expand_dims(x==self.padding, -1),self.ntoken,-1)

        out = np.zeros(x.shape + (self.ntoken,), dtype = int)
        out[all_idx(x, axis=2)] = 1
        out = torch.tensor(out).type(torch.FloatTensor)
        out[torch.tensor(ignore)] = 0
        return out

class ContextIndicator(nn.Module):
    """
    class to implement indicator function (i.e. one-hot encoding)
    with an additional dimension to indicate context (so, it's actually one-or-two-hot)
    ntoken should still be the number of tokens + 1 !!

    Caveat: this needs the FULL SEQUENCE, not a single time point. That's probably a bad
    thing, and I should fix it. TODO: make this work on single time points.
    """
    def __init__(self, ntoken, ninp, padding_idx=-1):
        super().__init__()
        self.ntoken = ntoken
        self.padding = padding_idx

    def __repr__(self):
        return "ContextIndicator(ntoken=%d, ninp=%d)"%(self.ntoken,self.ntoken)

    def forward(self, x):
        """
        Convert list of indices x to an indicator (i.e. one-hot) representation
        x is shape (lseq, ...)
        """

        def all_idx(idx, axis):
            """ from stack exchange"""
            grid = np.ogrid[tuple(map(slice, idx.shape))]
            grid.insert(axis, idx)
            return tuple(grid)

        # determine the context
        y = self.determine_context(x.detach().numpy())

        ignore = np.repeat(np.expand_dims(x==self.padding, -1),self.ntoken,-1)

        out = np.zeros(x.shape + (self.ntoken,), dtype = int)
        out[all_idx(x, axis=2)] = 1
        out[:,:,-1] += y
        out = torch.tensor(out).type(torch.FloatTensor)
        out[torch.tensor(ignore)] = 0
        return out

    def determine_context(self,x):
        """return the times at which a number is being repeated"""
        def find_reps(seq,mem):
            o = np.zeros((1,)+seq.shape[1:])
            r = np.diff(np.append(o,np.cumsum(seq==mem, axis=0).astype(int) % 2, axis=0), axis=0)<0
            return r
        # rep = [np.diff(np.cumsum(x==t, axis=1).astype(int) % 2, prepend=0)<0 for t in range(self.ntoken)]
        rep = [find_reps(x,t) for t in range(self.ntoken)]
        return np.any(rep, axis=0).astype(int)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# %% decoders
class LinearDecoder(object):
    """
    For getting the probability that each token is currently represented
    Right now all decoders are separate 
    """
    
    def __init__(self, rnn, P, clsfr):
        super(LinearDecoder,self).__init__()
        
        # ntoken = rnn.encoder.ntoken

        # self.ntoken = ntoken
        self.tokens = list(range(P+1))
        self.nhid = rnn.nhid
        self.padding = rnn.padding

        self.clsfr = clsfr
        
    def fit(self, H, I, explicit, t_=None, **cfargs):
        """
        Trains classifiers for each time bin
        optionally supply `t_` manually, e.g. if you want to train a single
        classifier over all time.
        """

        if t_ is None:
            t_ = np.cumsum(I!=self.padding, axis=0)-1 # binning of time

        t_lbs = np.unique(t_[t_>=0])

        L = int(np.ceil(len(t_lbs)/2))

        # make targets
        ans = np.zeros(I.shape + (len(self.tokens),)) # lenseq x nseq x ntoken
        for p in self.tokens:
            ans[:,:,p] =np.apply_along_axis(self.is_memory_active, 0, I, p)

        # define and train classifiers
        clf = [[self.clsfr(**cfargs) for _ in t_lbs] for _ in self.tokens]

        H_flat = H.transpose(1,0,2).reshape((self.nhid,-1))
        for t in t_lbs:
            idx = np.nonzero(t_==t)

            for p in self.tokens[:-1]:
                clf[p][t].fit(H[idx[0],:,idx[1]], ans[t_==t,p])

            if explicit:
                clf[-1][t].fit(H_flat.T, ans[:,:,-1].flatten())
            else:
                clf[-1][t].fit(H_flat.T, t_.flatten()>=L)

        # package
        coefs = np.zeros((len(self.tokens), self.nhid, len(t_lbs)))
        thrs = np.zeros((len(self.tokens), len(t_lbs)))
        for p in self.tokens:
            for t in t_lbs:
                coefs[p,:,t] = clf[p][t].coef_/la.norm(clf[p][t].coef_)
                thrs[p,t] = -clf[p][t].intercept_/la.norm(clf[p][t].coef_)

        self.clf = clf
        self.time_bins = t_lbs
        self.coefs = coefs
        self.thrs = thrs

    def test(self, H, I, t_=None):
        """Compute performance of the classifiers on dataset (H, I)"""

        # this is a bit tricky, because, in the most general setup, the model we use
        # at a given time & token depends on the particular sequence ... 

        if t_ is None:
            t_ = np.cumsum(I!=self.padding, axis=0)-1 # bin time

        if not (t_.max()<=self.time_bins.max() and t_.min()>=self.time_bins.min()):
            raise ValueError('The time bins of testing data are not '
                'the same as training data!\n'
                'Was trained on: %s \n'
                'Was given: %s'%(str(self.time_bins),str(np.unique(t_[t_>=0]))))

        ans = np.zeros(I.shape + (len(self.tokens),)) # lenseq x nseq x ntoken
        for p in self.tokens:
            ans[:,:,p] = np.apply_along_axis(self.is_memory_active, 0, I, p)

        # compute appropriate predictions
        C = (self.coefs[:,:,t_].transpose((0,2,1,3))*H[None,:,:,:]).sum(2)
        proj = C - self.thrs[:,t_]

        # evaluate
        perf = np.mean(ans.transpose((2,0,1)) == (proj>=0), axis=2)

        return perf

    def project(self, H, t_clf):
        """
        returns H projected onto classifiers, where `t_clf` gives the classifier index
        (in time bins) of the classifier to use at each time in each sequence.
        """

        C = (self.coefs[:,:,t_clf].transpose((0,2,1,3))*H[None,:,:,:]).sum(2)
        proj = C - self.thrs[:,t_clf]

        return proj

    def is_memory_active(self, seq, mem):
        """
        Returns step function for whether mem is 'active' in seq
        """
        stp = np.cumsum((seq==mem).astype(int)) % 2
        return stp
