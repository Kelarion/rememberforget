import getopt, sys
#sys.path.append(CODE_DIR)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import compress, permutations
#from joblib import Parallel, delayed
#import multiprocessing
import numpy as np
import scipy.special as spc
from students import RNNModel

#%%
def train_and_test(rnn_type, N, P, Ls, nseq, ntest, nlayers=1, pad=-1, 
                   expargs=None, dlargs=None, optargs=None,
                   criterion=None, alg=None, nepoch=2000, dead_time=None,
                   train_embedding=False, persistent=False, 
                   return_data=False, verbose=True):
    """
    function to reduce clutter 
    """
    
    AB = list(range(P))
    nneur = N             # size of recurrent network
    ninp = P              # dimension of RNN input
    
    if type(Ls) is not list:
        Ls = [Ls]
    
    ## DEFAULTS --------------------------------
    # sequence parameters
    if expargs is None: # parameters for `draw_seqs`
        expargs = {} # will propagate function defaults
        expargs['explicit'] = True
        expargs['be_picky'] = False
        if verbose:
            print('Using default sequence parameters'
                  'be_picky: false, context_cue: explicit')

    explicit = expargs['explicit']
    be_picky = expargs['be_picky']
    
    # optimisation parameters
    if alg is None:
        alg = optim.Adam
    if dlargs is None:
        dlargs = {'num_workers': 2, 
                  'batch_size': 64, 
                  'shuffle': True}  # dataloader arguments
    if optargs is None:
        optargs = {'lr': 5e-4}      # optimiser args
    if criterion is None:
        criterion = torch.nn.NLLLoss()
    # -------------------------------------------
    
    if explicit or persistent:
        ninp += 1
    
    # check this before proceding
    if be_picky and (math.factorial(max(Ls))/math.factorial(P-max(Ls)))>1e7:
        be_picky = False
        print("Hey! What are you doing!? That's too many sequences, my friend.")
    
    # train and test
    if verbose:
        print('TRAINING %s NETWORK' % str(Ls))
    
    nums, ans, numstest, anstest = make_dset(Ls, AB, int(nseq/len(Ls)), 
                                             ntest=int(ntest/len(Ls)),
                                             padding=pad,
                                             dead_time=dead_time,
                                             expargs=expargs)
#    nums[nums==-1] = pad
    
    ptnums = torch.tensor(nums).type(torch.LongTensor)
    ptans = torch.tensor(ans).type(torch.FloatTensor)
    
    # specify model and loss
    rnn = RNNModel(rnn_type, ninp, ninp, nneur, nlayers, 
                   embed=train_embedding, persistent=persistent,
                   padding=pad)
    
    loss_ = rnn.train(ptnums, ptans, optargs, dlargs, algo=alg, 
                      nepoch=nepoch, criterion=criterion,
                      do_print=False, epsilon=5e-4)
    
    if verbose:
        print('TESTING %s NETWORK' % str(Ls))
    
    if be_picky: # ensure that test set is different
        accuracy = test_net(rnn, AB, test_data=(numstest, anstest),
                            dead_time=dead_time, **expargs)
    else:
        accuracy = test_net(rnn, AB, Ls, ntest, dead_time=dead_time **expargs)
    
    if verbose:
        print('- '*20)
    
    if return_data:
        return loss_, accuracy, rnn, (nums, ans, numstest, anstest)
    else:
        return loss_, accuracy, rnn

def test_net(rnn, AB, these_L=None, ntest=None, test_data=None, padding=-1,
             verbose=False, return_hidden=False, give_gates=False, dead_time=None,
             **expargs):
    """
    Run `rnn` on `ntest` sequences of L in `these_L`, drawn from alphabet `AB` 
    Returns the accuracy on the test sequences, and optionally the hidden 
    activity+input sequence and/or gate activations. 
    
    If `ntest` isn't specified, you can instead give `test_data`, which is a 
    tuple of (seqs, ans).
    
    ONLY SET give_gates=True IF RNN IS OF CLASS tanh-GRU!! 
    
    Outputs in order: (accuracy, hidden, inputs, (update, reset))
    """
    
#    if (ntest is None) and (test_data is None):
#        raise(ValueError)
    
    if ntest is not None:
        test_nums, test_ans, _, _ = make_dset(these_L, AB, ntest, 0,
                                              padding=rnn.padding,
                                              dead_time=dead_time,
                                              expargs=expargs)
    else:
        test_nums, test_ans = test_data
    
    Ntest = test_nums.shape[0]

    ignore = torch.tensor(test_nums==rnn.padding)
    test_inp = torch.tensor(test_nums).type(torch.LongTensor)
#    print(test_inp.shape)
#    print(ignore.shape)
#    test_inp[ignore] = 0
    test_inp = test_inp.transpose(0,1) # lseq x ntest
    lseq = (~ignore).sum(1)
    t_final = -(np.fliplr(test_nums!=rnn.padding).argmax(1)+1) # which is the final timestep
    lmax = test_nums.shape[1]
    
    # test predictions
    if return_hidden:
        H = torch.zeros(test_nums.shape[1], rnn.nhid, Ntest)
    if give_gates:
        Z = torch.zeros(lmax, rnn.nhid, Ntest)
        R = torch.zeros(lmax, rnn.nhid, Ntest)
    
    hid = rnn.init_hidden(test_inp.shape[1])
    if return_hidden or give_gates:
        # because pytorch only returns hidden activity in the last time step,
        # we need to unroll it manually. 
        O = torch.zeros(lmax, Ntest, rnn.decoder.out_features)
        emb = rnn.encoder(test_inp)
        for t in range(lmax):
            if give_gates:
                out, hid, ZR = rnn.rnn(emb[t:t+1,...], hid, give_gates=True)
                Z[t,:,:] = ZR[0].squeeze(0).T
                R[t,:,:] = ZR[1].squeeze(0).T
            else:
                out, hid = rnn.rnn(emb[t:t+1,...], hid)
            dec = rnn.softmax(rnn.decoder(out))
            H[t,:,:] = hid.squeeze(0).T
            O[t,:,:] = dec.squeeze(0)
    else: # this is faster somehow
        O, _ = rnn(test_inp, hid)
        
    O = O.detach().numpy()
    O_T = O[t_final,np.arange(Ntest),:] # output at final timestep
    whichone = O_T.argmax(1) # network's answer

    accuracy = [np.mean(whichone[lseq==l] == test_ans[lseq==l]) for l in np.unique(lseq)]
    
    # collect for output
    outputs = (accuracy, )
    if return_hidden:
        H = H.detach().numpy()
        outputs += (H, test_nums.T)
    if give_gates:
        outputs += (Z.detach().numpy(), R.detach().numpy())
    
    return outputs

#%%
def make_dset(Ls, AB, nseq, ntest, padding=-1, dead_time=None, expargs={}):
    """
    Wrapper for draw_seqs, which concatenates sequences of multiple lengths into
    a single dataset. Fills extra space with `padding`, which you should keep
    track of when you feed this data into an RNN. Specifically, the value of 
    `padding` should be the same as `padding` in the RNNModel.train method.
    (The defaults in both functions are matched, so don't worry about it.)
    
    If `be_picky=True`, then train and test sets are disjoint -- but heed the 
    warning in the `draw_seqs` help text.
    
    `dead_time` specifies the total amount of time spent after tokens are
    presented -- i.e. how much time to distribute across 
    
    returns seqs, ans, test_seqs, test_ans
    """
    lseq_max = 2*max(Ls)

#    dset_nums = []
#    dset_nums_test = []
#    dset_ans = []
#    dset_ans_test = []
    dset_nums = np.zeros((0,lseq_max))
    dset_nums_test = np.zeros((0,lseq_max))
    dset_ans = np.zeros(0)
    dset_ans_test = np.zeros(0)
    
    for l in Ls:
        nums, ans = draw_seqs(l, nseq+ntest, Om=AB, **expargs)
#        print('drawn')
        if nums.shape[0] < nseq+ntest:
            Nseq = int(nseq*nums.shape[0]/(nseq+ntest))
            Ntest = nums.shape[0]-Nseq
        else:
            Nseq = nseq
            Ntest = ntest
        
        trains = np.random.choice(Nseq+Ntest, Nseq, replace=False)
        tests = np.setdiff1d(np.arange(Nseq+Ntest), trains)
    
        train_seqs = np.pad(nums[trains,:], ((0,0),(0,lseq_max-nums.shape[1])),
                            'constant', constant_values=padding)
        test_seqs = np.pad(nums[tests,:], ((0,0),(0,lseq_max-nums.shape[1])),
                           'constant', constant_values=padding)
#        print('padded')
#        if Ls.index(l) == 0:
#            dset_nums = train_seqs
#            dset_ans = ans[trains]
#            dset_nums_test = test_seqs
#            dset_ans_test = ans[tests]
#        else: 
        dset_nums = np.append(dset_nums, train_seqs, axis=0)
        dset_ans = np.append(dset_ans, ans[trains])
        dset_nums_test = np.append(dset_nums_test, test_seqs, axis=0)
        dset_ans_test = np.append(dset_ans_test, ans[tests])
        
#        dset_nums += train_seqs.tolist()
#        dset_ans += ans[trains].tolist()
#        
#        dset_nums_test += test_seqs.tolist()
#        dset_ans_test += ans[tests].tolist()
    
    dset_nums = dset_nums[:, ~np.all(dset_nums==padding, axis=0)]
    dset_nums_test = dset_nums_test[:, ~np.all(dset_nums_test==padding, axis=0)]
    
    if dead_time is not None:
        nT = dset_nums.shape[1] + dead_time
        
        _dset_nums = np.ones((dset_nums.shape[0],nT))*padding
        idx = np.random.rand(dset_nums.shape[0],nT).argsort(1)[:,:(nT-dead_time)]
        idx = np.sort(idx,1)
        idx[:,0] = 0
#        print(idx)
        np.put_along_axis(_dset_nums, idx, dset_nums, axis=1)
        dset_nums = _dset_nums
        
        if dset_nums_test.shape[0] > 0:
            _dset_nums_test = np.ones((dset_nums_test.shape[0],nT))*padding
            idx = np.random.rand(dset_nums_test.shape[0],nT).argsort(1)[:,:(nT-dead_time)]
            idx = np.sort(idx,1)
            idx[:,0] = 0
            np.put_along_axis(_dset_nums_test, idx, dset_nums_test, axis=1)
            dset_nums_test = _dset_nums_test
    
    return dset_nums, dset_ans, dset_nums_test, dset_ans_test

def draw_seqs(L, N, Om, mirror=False, explicit=True, be_picky=False,
              signal_tokens=None):
    """
    Draw N sequences of length 2L, such that L-1 elements occur twice, and
    all elements are drawn from Om.
    
    If be_picky=True, it'll return only unique sequences (if few enough)
        - if mirrored, this is at most L*|Om|!/(|Om|-L)!
        - if we allow repeats in any order, it is L!|Om|!/(|Om|-L)!
        - please please please consider how many sequences you're generating
            * at some point, we need to index the set of all possible 
              permutations of Om of length L, so that gets big quickly
            * roughly: don't be_picky if |Om|!/(|Om|-L)! > ~10^7
    
    Supplying `signal_tokens` will further restrict the sequences to those with
    non-repeated tokens that are members of signal_tokens. For example, if 
    signal_tokens = [0,1] then only 0 or 1 will be non-repeated. Right now, 
    this only works for be_picky=True, and takes substantially longer. In any
    case, this will severely restrict the number of unique sequences available.
    
    TODO: 
        Currently the repetitions are all at the end -- maybe we'll change this
    """
    if signal_tokens is None:
        signal_tokens = Om

    if explicit:
        forget_switch = len(Om)
    else:
        forget_switch= None
    
    P = len(Om)
    X = len(signal_tokens)
#    print(X)
    if be_picky:
        maxseq = unique_seqs(P, L, X, mirror)
        if N > maxseq: # if P and L are sufficiently small
            N = maxseq
        
        # select a random subset of all permutations of length L (or all of them)
        # (we'll take different strategies for subsampling and exhaustive sampling)
        nperm_tot = math.factorial(P)/math.factorial(P-L)
        nperm = round(pintersection(P,L,X)*nperm_tot) # number of permutations
        if N == maxseq:
            npfx = nperm
        else:
            if X != P:
                # compute expected number of sequences per prefix (conditionally)
#                nx = sum([n*pintersection(P,L,X,n) for n in range(1,X+1)])/pintersection(P,L,X)
                # get minimum possible number of 
                nx = min([n for n in range(1,X+1) if pintersection(P,L,X,n)>0])
                npfx = np.ceil(N/nx)
            else:
                npfx = np.ceil(N/L) # number of 'prefixes'

#        def checkifin(x,signal):
#            return np.any(np.isin(x, signal))
        if X != P:
            checkifin = lambda x:np.any(np.isin(x, range(X)))
            perms = filter(checkifin,permutations(Om, L)) # there are many of these
        else:
            perms = permutations(Om, L)
        # select a random subset of permutations
        pinds = np.random.permutation(np.arange(nperm)<npfx) # shuffled binary indexing
        toks = np.array(list(compress(perms, pinds))) # select random elements of permutations
        
        # choose the non-repeated tokens
        inds = (np.tile(np.eye(L), int(min([nperm,npfx]))).T > 0)

        if X != P:
            sig_toks = np.isin(toks, range(X)) # ensure we only select 'signal' tokens
            inds = inds[sig_toks.flatten(),:] # as the non-repeated ones
        
            toks = np.repeat(toks, sig_toks.sum(1), axis=0)
        else:
            toks = np.repeat(toks, L, axis=0)
        
        A = toks[inds]
        skot = toks[inds==0].reshape((-1, L-1))
        if npfx >= nperm: # i.e. we need more 
            # add more sequences by permuting the forget phase
            nsubperm = math.factorial(L-1)
            d = int(np.ceil(N/(L*nperm))) # how many permutations to use (always <= (L-1)!)
            
            toks = np.repeat(toks, d, axis=0)
            A = np.repeat(A, math.factorial(L-1), axis=0)
            
            sigma = np.random.permutation # for convenience
            allperms = lambda x:list(compress(permutations(x, L-1),\
                                              sigma(np.arange(nsubperm)<d)))
            
            skot = np.apply_along_axis(allperms, 1, skot).reshape((-1, L-1))
        else:
            if mirror: # get repeated tokens
                skot = np.fliplr(skot) 
            else:
                skot = scramble(skot, axis=0)
        
        if forget_switch is not None:
            skot = np.insert(skot, 0, forget_switch, axis=1)

        S = np.concatenate((toks, skot), axis=1)

        # scramble
        shf = np.random.choice(S.shape[0], int(N), replace=False)
        S = S[shf,:]
        A = A[shf]
        
    else: # if there are many unique sequences, just take random samples
        if forget_switch is not None:
            S = np.zeros((N, 2*L), dtype = int)
        else:
            S = np.zeros((N, 2*L-1), dtype = int)
        A = np.zeros((N,1), dtype = int)
        
        # draw tokens from alphabet
        for n in range(N):
            toks = np.random.choice(Om, L, replace=False)
            distok = np.random.choice(L,1)
            skot = np.flip(np.delete(toks,distok))
            if not mirror:
                skot = skot[np.random.choice(L-1,L-1,replace=False)]
            if forget_switch is not None:
                skot = np.append(forget_switch, skot)
            S[n,:] = np.append(toks,skot)
            A[n] = toks[distok]
            
    return S, A

#%% file I/O
def expfolders(exp, pers, emb):
    """Folder hierarchy for organising results
        - expfolders(explicit?, persistent?, embed?)"""
    folds = 'results/'
    if exp:
        folds += 'explicit/'
    else:
        folds += 'implicit/'
    if pers:
        folds += 'persistent/'
    if emb:
        folds += 'embedded/'
    return folds

def suffix_maker(N, P, rnn_type, Q=None, dead_time=None):
    """
    A consistent naming scheme to make it, supposedly, easier to find outputs
    of experiments.
    
    Will give something like 'X_Y_Z' to be added before the file extension.
    e.g., I might do 
        `filename = 'accuracy' + suffix_maker(10, 10, 'GRU') + '.npy'
        print(filename)`
    which would output
        `accuracy_10_10_GRU.npy`
        
    Can add further things whenever you want, but for compatibility, try to 
    keep the `_N_P_rnn` kernel intact -- unless it demonstrably sucks.
    """
    suff = '_%d_%d_%s'%(N, P, rnn_type)
    if Q is not None:
        suff += '_%d'%Q
    if dead_time is not None:
        suff += '_plus%d'%dead_time
    
    return suff

##%% plotting
#def myplot(cmap=None, cols==None, **mplargs):
#    
#    
#    plt.plot(**mplargs)

#%% helpers
def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    based on unutbu's answer on stack exchange
    """ 
    b = a.swapaxes(axis, -1)
    idx = np.random.rand(*b.shape).argsort(0)
    b = np.take_along_axis(b, idx, 0)
    return b.swapaxes(axis, -1)    

def is_memory_active(seq, mem):
    """
    Returns step function for whether mem is 'active' in seq
    """
    stp = np.cumsum((seq==mem).astype(int)) % 2
    return stp

def unique_seqs(P, L, X=None, mirror = False):
    """
    Return number of unique sequences 
    """
    if mirror:
        maxseq = int(L*math.factorial(P)/math.factorial(P-L))
    else:
        if X is not None:
            maxseq = math.factorial(P)/math.factorial(P-L)
            maxseq *= sum([n*pintersection(P,L,X,n) for n in range(1,X+1)])
            maxseq = int(maxseq)
        else:
            maxseq = int(math.factorial(L)*math.factorial(P)/math.factorial(P-L))
    return maxseq

def pintersection(P,L,X,n=None):
    """
    Say there is a set with P elements, and a subset of it with X < P elements;
    this computes the probability that another subset with L elements shares n 
    elements with the first subset. If n is unspecified, returns prob(n > 0).
    """
    if n is None:
        return 1-spc.binom(P-X,L)/spc.binom(P,L)
    else:
        return spc.binom(P,n)*spc.binom(P-X,L-n)*spc.binom(P-n,X-n)/(spc.binom(P,L)*spc.binom(P,X))


