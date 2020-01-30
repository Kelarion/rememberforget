"""
Classes used in the Remember-Forget experiments.
 
Includes:
    - RNNModel: class that currently supports {'LSTM', 'GRU', 'tanh', 'relu','tanh-GRU', 'relu-GRU'}
    recurrent neural networks. includes a save, load, and train method.
    - stateDecoder: does the memory decoding. includes a train method.

"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import scipy.linalg as la
import scipy.special as spc
# from needful_functions import is_memory_active

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
        # self.softmax = nn.LogSoftmax(dim=2)
        
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

        # decoded = self.softmax(self.decoder(output))
        decoded = self.decoder(output)

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

    def transparent_foward(self, input, hidden, give_gates=False, debug=False):
        """
        Run the RNNs forward function, but returning hidden activity throughout the sequence

        it's slower than regular forward, but often necessary
        """

        lseq, nseq = input.shape
        ispad = (input == self.padding)

        H = torch.zeros(lseq, self.nhid, nseq)
        if give_gates:
            Z = torch.zeros(lseq, self.nhid, nseq)
            R = torch.zeros(lseq, self.nhid, nseq)
        
        # because pytorch only returns hidden activity in the last time step,
        # we need to unroll it manually. 
        O = torch.zeros(lseq, nseq, self.decoder.out_features)
        emb = self.encoder(input)
        for t in range(lseq):
            if give_gates:
                out, hidden, ZR = self.rnn(emb[t:t+1,...], hidden, give_gates=True)
                Z[t,:,:] = ZR[0].squeeze(0).T
                R[t,:,:] = ZR[1].squeeze(0).T
            else:
                out, hidden = self.rnn(emb[t:t+1,...], hidden)
            dec = self.decoder(out)
            naan = torch.ones(hidden.squeeze(0).shape)*np.nan
            H[t,:,:] = torch.where(~ispad[t:t+1,:].T, hidden.squeeze(0), naan).T
            O[t,:,:] = dec.squeeze(0)

        if give_gates:
            if debug:
                return dec, H, Z, R, emb
            else:
                return dec, H, Z, R
        else:
            if debug:
                return dec, H, emb
            else:
                return dec, H

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
        criterion=nn.CrossEntropyLoss(), nepoch=1000, do_print=True,
        epsilon=0, test_data=None, save_params=False):
        """
        Train rnn on data X and labels Y (both torch tensors).
        X, Y need to have samples as the FIRST dimension

        supply test data as (X,Y), optionally
        """

        if type(criterion) is torch.nn.modules.loss.BCEWithLogitsLoss:
            self.q_ = int(torch.sum(criterion.weight>0))
        else:
            self.q_ = self.decoder.out_features
        
        self.optimizer = algo(self.parameters(), **optparams)
        padding = self.padding

        dset = torch.utils.data.TensorDataset(X, Y)
        trainloader = torch.utils.data.DataLoader(dset, **dlparams)

        self.init_metrics()

        # loss_ = np.zeros(0)
        # test_loss_ = np.zeros(0)
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

                # we're going to train on the final non-padding time point
                t_final = -(np.flipud(btch!=self.padding).argmax(0)+1) # index of final time point
                # btch[btch==self.padding] = 0 # set to some arbitrary value

                out, hidden = self(btch, hidden)
                output = out[t_final, np.arange(btch.size(1)), :]

                loss = criterion(output.squeeze(0),labs.squeeze()) # | || || |_
                loss.backward()

                self.optimizer.step()

                # update loss
                running_loss += loss.item() 
                self.metrics['train_loss'] = np.append(self.metrics['train_loss'], loss.item())
            
            # compute the metrics at each epoch of learning
            idx = np.random.choice(X.size(0),np.min([1500, X.size(0)]), replace=False)
            self.compute_metrics((X[idx,:].T, Y[idx]), test_data, criterion)

            if save_params:
                thisdir = '/home/matteo/Documents/github/rememberforget/results/justremember/'
                self.save(thisdir+'params_epoch%d.pt'%epoch)

            #print loss
            if (epsilon>0) and (np.abs(running_loss-prev_loss) <= epsilon):
                print('~'*5)
                print('[%d] Converged at loss = %0.3f'%(epoch+1, running_loss/(i+1)))
                return
                # return loss_, metrics
            # print to screen
            if do_print:
                print('[%d] loss: %.3f' %
                      (epoch + 1, running_loss / i))
                running_loss = 0.0
                prev_loss = running_loss

        print('[%d] Finished at loss = %0.3f'%(epoch+1, running_loss/(i+1)))
        print('~'*5)
        # return loss_

    # for computing metrics
    def init_metrics(self):
        """
        Initialise the various metrics to be computed during learning
        """

        self.metrics = {}

        self.metrics['train_loss'] = np.zeros(0)
        self.metrics['test_loss'] = np.zeros(0)

        # self.orth_clf = LinearDecoder(self, self.q_, MeanClassifier)
        # self.metrics['train_orthogonality'] = np.zeros(0)
        # self.metrics['test_orthogonality'] = np.zeros(0)

        self.metrics['train_parallelism'] = np.zeros((0,self.q_)) 
        self.metrics['test_parallelism'] = np.zeros((0,self.q_))

    def compute_metrics(self, train_data, test_data, criterion):
        """
        Compute the various metrics on the train and test data. Can be done at any point in training.
        """
        m = self.metrics
        warnings.filterwarnings('ignore','Mean of empty slice')

        ## load data
        trn, trn_labs = train_data
        tst, tst_labs = test_data

        # trn = trn.transpose(1,0)
        tst = tst.transpose(1,0)

        t_final = -(np.flipud(trn!=self.padding).argmax(0)+1)
        test_tfinal = -(np.flipud(tst!=self.padding).argmax(0)+1)

        ntest = tst.size(1)
        P = self.decoder.out_features

        ## training data ###########################################################
        hidden = self.init_hidden(trn.size(1))
        out, hidden = self.transparent_foward(trn, hidden)
        # output = out[t_final, np.arange(trn.size(1)), :]
        output = out.squeeze()
        # compute orthogonality
        mem_act = np.array([np.cumsum(trn==p,axis=0).int().detach().numpy() % 2 \
            for p in range(self.q_)]).transpose((1,2,0))

        ps_clf = LinearDecoder(self, 2**(self.q_-1), MeanClassifier)
        ps = []
        for d in Dichotomies(mem_act, 'simple'):
            np.warnings.filterwarnings('ignore',message='invalid value encountered in')
            ps_clf.fit(hidden.detach().numpy(), d)
            new_ps = ps_clf.orthogonality()
            ps.append(new_ps)
            # if new_ps > ps:
            #     ps = new_ps
        m['train_parallelism'] = np.append(m['train_parallelism'], np.array(ps).T, axis=0)

        # print(mem_act.shape)
        # print(hidden.shape)
        # self.orth_clf.fit(hidden.detach().numpy(), mem_act)
        # orth_score = self.orth_clf.orthogonality()
        # m['train_orthogonality'] = np.append(m['train_orthogonality'], orth_score)

        ## test data ##############################################################
        hidden = self.init_hidden(tst.size(1))
        out, hidden = self.transparent_foward(tst, hidden)
        output = out.squeeze()
        # print(hidden[-1,:,:])
        # output = out[test_tfinal, np.arange(tst.size(1)), :]
        # raise Exception

        # compute loss
        test_loss = criterion(output.squeeze(0),tst_labs.squeeze())

        m['test_loss'] = np.append(m['test_loss'], test_loss.item())

        # compute orthogonality
        mem_act = np.array([np.cumsum(tst==p,axis=0).int().detach().numpy() % 2 \
            for p in range(self.q_)]).transpose((1,2,0))

        # self.orth_clf.fit(hidden.detach().numpy(), mem_act)
        # orth_score = self.orth_clf.orthogonality()
        # m['test_orthogonality'] = np.append(m['test_orthogonality'], orth_score)

        # compute parallelism
        ps_clf = LinearDecoder(self, 2**(self.q_-1), MeanClassifier)
        ps = []
        for d in Dichotomies(mem_act, 'simple'):
            np.warnings.filterwarnings('ignore',message='invalid value encountered in')
            ps_clf.fit(hidden.detach().numpy(), d)
            new_ps = ps_clf.orthogonality()
            ps.append(new_ps)
            # if new_ps > ps:
            #     ps = new_ps
        m['test_parallelism'] = np.append(m['test_parallelism'], np.array(ps).T, axis=0)

        ## package #################################################################
        self.metrics = m
        warnings.filterwarnings('default')

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
        self.tokens = list(range(P))
        self.nhid = rnn.nhid
        self.padding = rnn.padding

        self.clsfr = clsfr

    def __repr__(self):
        return "LinearDecoder(ntoken=%d, classifier=%s)"%(len(self.tokens),self.clsfr)

    def fit(self, H, labels, t_=None, **cfargs):
        """
        Trains classifiers for each time bin
        optionally supply `t_` manually, e.g. if you want to train a single
        classifier over all time.
        """

        if t_ is None:
            # t_ = np.cumsum(I!=self.padding, axis=0)-1 # binning of time
            # t_ = (np.cumsum(np.ones((H.shape[0], H.shape[2])), axis=0)-1).astype(int)
            t_ = np.zeros((H.shape[0], H.shape[2]), dtype=int)

        t_lbs = np.unique(t_[t_>=0])

        # L = int(np.ceil(len(t_lbs)/2))

        # # make targets
        # labels = np.zeros(I.shape + (len(self.tokens),)) # lenseq x nseq x ntoken
        # for p in self.tokens:
        #     labels[:,:,p] =np.apply_along_axis(self.is_memory_active, 0, I, p)

        # define and train classifiers
        clf = [[self.clsfr(**cfargs) for _ in t_lbs] for _ in self.tokens]

        H_flat = H.transpose(1,0,2).reshape((self.nhid,-1))
        for t in t_lbs:
            idx = np.nonzero(t_==t)
            for p in self.tokens:
                clf[p][t].fit(H[idx[0],:,idx[1]], labels[t_==t,p])
            # clf[-1][t].fit(H_flat.T, t_.flatten()>=L) # last decoder is always context

        # package
        coefs = np.zeros((len(self.tokens), self.nhid, len(t_lbs)))*np.nan
        thrs = np.zeros((len(self.tokens), len(t_lbs)))*np.nan
        for p in self.tokens:
            for t in t_lbs:
                if ~np.all(np.isnan(clf[p][t].coef_)):
                    coefs[p,:,t] = clf[p][t].coef_/la.norm(clf[p][t].coef_)
                    thrs[p,t] = -clf[p][t].intercept_/la.norm(clf[p][t].coef_)

        self.clf = clf
        self.nclfr = len(self.tokens)
        self.time_bins = t_lbs
        self.coefs = coefs
        self.thrs = thrs

    def test(self, H, labels, t_=None):
        """Compute performance of the classifiers on dataset (H, I)"""

        # this is a bit tricky, because, in the most general setup, the model we use
        # at a given time & token depends on the particular sequence ... 

        if t_ is None:
            # t_ = np.cumsum(I!=self.padding, axis=0)-1 # binning of time
            # t_ = (np.cumsum(np.ones((H.shape[0], H.shape[2])), axis=0)-1).astype(int)
            t_ = np.zeros((H.shape[0], H.shape[2]), dtype=int)

        if not (t_.max()<=self.time_bins.max() and t_.min()>=self.time_bins.min()):
            raise ValueError('The time bins of testing data are not '
                'the same as training data!\n'
                'Was trained on: %s \n'
                'Was given: %s'%(str(self.time_bins),str(np.unique(t_[t_>=0]))))

        # compute appropriate predictions
        proj = self.project(H, t_)

        # evaluate
        correct = labels.transpose((2,0,1)) == (proj>=0)
        perf = np.array([correct[:,t_==i].mean(1) for i in np.unique(t_)]).T

        return perf

    def margin(self, H, labels, t_=None):
        """ 
        Compute classification margin of each classifier, defined as the minimum distance between
        any point and the classification boundary.

        assumes class labels are 0 or 1
        """

        if t_ is None:
            # t_ = np.cumsum(I!=self.padding, axis=0)-1 # binning of time
            t_ = (np.cumsum(np.ones((H.shape[0], H.shape[2])), axis=0)-1).astype(int)

        # compute predictions
        proj = self.project(H, t_)
        dist = proj*(2*labels.transpose((2,0,1))-1) # distance to boundary

        marg = np.array([[dist[p, t_==t].min() for t in np.unique(t_)] \
            for p in self.tokens])

        return marg

    def project(self, H, t_clf):
        """
        returns H projected onto classifiers, where `t_clf` gives the classifier index
        (in time bins) of the classifier to use at each time in each sequence.
        """

        C = (self.coefs[:,:,t_clf].transpose((0,2,1,3))*H[None,:,:,:]).sum(2)
        proj = C - self.thrs[:,t_clf]

        return proj

    def orthogonality(self, which_clf=None):
        """
        Computes the average dot product.
        """
        
        if which_clf is None:
            # which_clf = np.ones(self.nclfr)>0
            which_clf = ~np.any(np.isnan(self.coefs),axis=(1,2))

        C = self.coefs[which_clf, ...]
        # C = np.where(np.isnan(C), 0, C)
        csin = (np.einsum('ik...,jk...->ij...', C, C))
        PS = np.triu(csin.transpose(2,1,0),1).sum((1,2))
        PS /= sum(which_clf)*(sum(which_clf)-1)/2
        
        return PS

class MeanClassifier(object):
    """
    A class which just computes the vector between the mean of two classes. Is used for
    computing the parallelism score, for a particular choice of dichotomy.
    """
    def __init__(self):
        super(MeanClassifier,self).__init__()

    def fit(self, X, Y):
        """
        X is of shape (N_sample x N_dim), Y is (N_sample,) binary labels
        """

        V = np.nanmean(X[Y>0,:],0)-np.nanmean(X[Y<=0,:],0)

        self.coef_ = V
        if np.all(np.isnan(V)):
            self.intercept_ = np.nan
        else:
            self.intercept_ = la.norm(V)/2 # arbitrary intercept

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# helper classes

class Dichotomies:
    """An iterator for looping over all dichotomies when computing parallelism"""

    def __init__(self, labels, dictype='simple'):
        """
        Takes an n-bit binary vector for each datapoint, and produces an iterator over the possible
        dichotomies of those classes. Each 'dichotomy' of the iterator is a re-labeling of the data
        into 2^(n-1) binary classifications.

        More specifically, 'labels' should be a binary vector for each datapoint you want to compute
        parallelism with. e.g. each timepoint in a sequence. 
        
        labels should be shape (..., n) where n is the number of binary variables.

        TODO: support various types of dichotomy -- right now only considers dichotomies where the
        same binary variable is flipped in every class (calling it a 'simple' dichotomy).
        """

        self.labels = labels
        if dictype == 'simple':
            self.ntot = labels.shape[-1] # number of binary variables

            # 'cond' is the lexicographic enumeration of each binary condition
            # i.e. converting each ntot-bit binary vector into decimal number
            self.cond = np.einsum('...i,i',self.labels,2**np.arange(self.ntot))
        else:
            raise ValueError('Value for "dictype" is not valid: ' + dictype)

        self.dictype = dictype

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr < self.ntot:
            if self.dictype == 'simple':
                p = self.curr
                # we want g_neg, the set of points in which condition p is zero
                bit_is_zero = np.arange(2**self.ntot)&np.array(2**p) == 0
                g_neg = np.arange(2**self.ntot)[bit_is_zero]

                L = np.array([np.where((self.cond==n)|(self.cond==n+(2**p)),self.cond==n,np.nan)\
                    for n in g_neg])

            self.curr +=1

            return L.transpose(1,2,0)
        else:
            raise StopIteration

