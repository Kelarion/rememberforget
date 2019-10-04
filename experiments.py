"""
Remember-Forget experiments

Take sequences of tokens drawn from an alphabet, return at the end of the sequence
the token that wasn't repeated. 

Notes
 - one interesting thing: a linear decoder works well at the last time step, but
 not at intermediate time steps; so it seems like the representation is changing
 during the task
 - training different linear decoder for each time step works better than one
 - decision tree works well
 - try using output network to decode at each timestep
 - look at the dimensionality of RNN activity throughout training?
"""

#%%
CODE_DIR = r'/home/matteo/Documents/github/rememberforget/'
svdir = r'~/Documents/uni/columbia/sueyeon/experiments/'

import sys
sys.path.append(CODE_DIR)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import compress, permutations
import matplotlib as mpl
import matplotlib.style as stl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.special as spc
from cycler import cycler
from students import RNNModel, stateDecoder

# change the garbage illegible default color cycle
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
    
#%% helpers
# basic version of the task
def draw_seqs(L, N, Om=10, switch=-1, be_picky=True, mirror=False):
    """
    Draw N sequences of length 2L-1, such that L-1 elements occur twice
    Optionally specify Om s.t. t_1, ..., t_L \in Om (i.e. |Om| >= L)
    If be_picky=True, it'll return only unique sequences (if few enough)
        - if mirrored, this is at most L*|Om|!/(|Om|-L)!
        - if we allow repeats in any order, it is L!|Om|!/(|Om|-L)!
    
    ToDo: 
        Currently the repetitions are all at the end -- maybe we'll change this
    """
    
    if mirror:
        maxseq = int(L*math.factorial(len(Om))/math.factorial(len(Om)-L))
    else:
        maxseq = int(math.factorial(L)*math.factorial(len(Om))/math.factorial(len(Om)-L))
        
    if be_picky:
        if N > maxseq: #unlikely
            N = maxseq
        
        # select a random subset of all L-length permutations
        npfx = np.ceil(N/L)
        nperm = math.factorial(len(Om))/math.factorial(len(Om)-L)
        
        perms = permutations(Om, L) # there are many of these
        pinds = np.random.permutation(np.arange(nperm)<npfx)
        toks = np.array(list(compress(perms, pinds))) # select random elements of permutations

        toks = np.repeat(toks, L, axis=0)
        # choose the non-repeated tokens
        inds = (np.tile(np.eye(L), int(npfx)).T > 0)
        
        if mirror: # get repeated tokens
            skot = np.fliplr(toks[inds==0].reshape((-1, L-1))) 
        else:
            skot = scramble(toks[inds==0].reshape((-1, L-1)), axis=0)
        if switch is not None:
            skot = np.insert(skot, 0, switch, axis=1)
            
        S = np.concatenate((toks, skot), axis=1)
        A = toks[inds]
        # scramble
        shf = np.random.choice(int(npfx*L), int(N), replace=False)
        S = S[shf,:]
        A = A[shf]
        
    else: # if there are many unique sequences, just take random samples
        if switch is not None:
            S = np.zeros((N, 2*L), dtype = int)
        else:
            S = np.zeros((N, 2*L-1), dtype = int)
        A = np.zeros((N,1), dtype = int)
        
        # draw tokens from alphabet
        for n in range(N):
            toks = np.random.choice(Om, L, replace = False)
            distok = np.random.choice(L,1)
            skot = np.flip(np.delete(toks,distok))
            if not mirror:
                skot = skot[np.random.choice(L-1,L-1,replace=False)]
            if switch is not None:
                skot = np.append(switch, skot)
            S[n,:] = np.append(toks,skot)
            A[n] = toks[distok]
            
    return S, A

# this is what I thought originally -- will save for the future
def nonrepeated(x, ntok):
    """ 
    the function that our network is learning 
    assumes time is along the axis=1 dimension
    returns non-repeated numbers in a logical representation
    """
    # find which numbers were repeated
    srtd = np.sort(x, axis = 1)
    shp = (x.shape[0], 1) + x[0,0,...].shape # size of non-time dimensions
    o = np.zeros(shp) > 0
    repeated = np.concatenate((o, (np.diff(srtd, axis=1) == 0)), axis =1)
    srtd[np.logical_not(repeated)] = -1
    
    # make array where time is replaced by unique tokens
    unq = np.einsum('ij,ij...->ij...',np.arange(ntok+1)[np.newaxis,:], np.ones(shp))
    
    # find which non-repeated numbers were present in each sequence
    unqtmp = unq.reshape((-1,ntok+1))
    srtdtmp = srtd.reshape((-1,srtd.shape[1]))
    xtmp = x.reshape((-1,x.shape[1]))
    numseq = srtdtmp.shape[0]
    isrep = np.array([np.isin(unqtmp[i,:],srtdtmp[i,:]) for i in range(numseq)])
    inseq = np.array([np.isin(unqtmp[i,:],xtmp[i,:]) for i in range(numseq)])
    nonrep = np.logical_not(isrep) & inseq
    
    # reshape
    nonrep = nonrep.reshape(unq.shape).astype(int)
    
    return nonrep

# miscellaneous
def as_indicator(x, Om):
    """
    Convert list of sequences x to an indicator (i.e. one-hot) representation
    extra dimensions are added at the end
    """
    def all_idx(idx, axis):
        """ from stack exchange"""
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)
    
    out = np.zeros(x.shape + (len(Om),), dtype = int)
    out[all_idx(x, axis=2)] = 1
    return out

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
    stp = np.cumsum((test_num==mem).astype(int)) % 2
    return stp

def unique_seqs(AB, L, mirror = False):
    """
    Return number of unique sequences 
    """
    if mirror:
        maxseq = int(L*math.factorial(len(AB))/math.factorial(len(AB)-L))
    else:
        maxseq = int(math.factorial(L)*math.factorial(len(AB))/math.factorial(len(AB)-L))
    return maxseq

def make_dset(Ls, AB, nseq, ntest, forget_switch, padding=0, be_picky=False):
    """
    function to reduce clutter below
    """
    lseq_max = 2*max(Ls)-1
    if forget_switch is not None:
        lseq_max += 1
        
    dset_nums = np.zeros((0,lseq_max))
    dset_nums_test = [[] for _ in range(len(AB))]
    dset_ans = np.zeros(0)
    dset_ans_test = [[] for _ in range(len(AB))]
    
    for l in Ls:
        nums, ans = draw_seqs(l, nseq+ntest, Om=AB, switch=forget_switch,
                              be_picky=be_picky)
        
        trains = np.random.choice(nseq+ntest, nseq, replace=False)
        tests = np.setdiff1d(np.arange(nseq+ntest), trains)
        
        nums_train = nums[trains,:]
        
        foo = np.pad(nums_train, ((0,0),(0,lseq_max-nums.shape[1])), 
                     'constant', constant_values=padding)
        dset_nums = np.append(dset_nums, foo, axis=0)
        dset_ans = np.append(dset_ans, ans[trains])
        
        dset_nums_test[l] = nums[tests,:]
        dset_ans_test[l] = ans[tests]
    
    return dset_nums, dset_ans, dset_nums_test, dset_ans_test


#%% Individual model: specify 
L = 5 # number of tokens
AB = list(range(10)) # alphabet
forget_switch = -1
nneur = 15 # size of recurrent network
nlayers = 1 # number of recurrent networks
nseq = int(np.min([5000, unique_seqs(AB, L)])) # number of sequences
ntest = np.ceil(nseq*0.1)
nseq -= ntest
rnn_type = 'GRU'

ninp = len(AB)
lenseq = 2*L-1
if forget_switch is not None:
    lenseq += 1
    ninp += 1
    AB_ = AB+[forget_switch] # extended alphabet
else:
    AB_ = AB

# draw data & targets
nums, ans = draw_seqs(L, nseq+ntest, Om=AB, switch=forget_switch)
trains = np.random.choice(nseq+ntest, nseq, replace=False)
tests = np.setdiff1d(np.arange(nseq+ntest), trains)

enc = nums[trains,:]
#enc = as_indicator(nums[trains,:], AB_).transpose(1,0,2) # should be (lenseq, nseq, ninp)
ptnums = torch.tensor(enc).type(torch.FloatTensor) # convert to pytorch
ptans = torch.tensor(ans[trains]).type(torch.FloatTensor)

#nums = np.random.multinomial(1, np.ones(ntok)/ntok, (lenseq, nseq))
#nums = np.random.choice(ntok, (lenseq,nseq))
#ans = nonrepeated(nums.T, ntok)

# specify model and loss
rnn = RNNModel(rnn_type, len(AB), ninp, nneur, nlayers)
criterion = torch.nn.NLLLoss()

#%% Individual model: train 
nepoch = 500 # how many times to go over data
alg = optim.Adam
dlargs = {'num_workers': 2, # dataloader arguments
          'batch_size': 64, 
          'shuffle': True}
optargs = {'lr': 1e-3} # optimiser args

loss_ = rnn.train(ptnums, ptans, optargs, dlargs, algo=alg, nepoch = nepoch)

#%% Generalisation 
# model and task specification
AB = list(range(10))        # alphabet
forget_switch =  max(AB)+1  # explicit vs 
#forget_switch =  None       # implicit
nneur = 3                  # size of recurrent network
nlayers = 1                 # number of recurrent networks
nseq_max = 5000             # number of training sequences
test_prop = 0.1             # proportion of test sequences
rnn_type = 'GRU'
be_picky = True             # should we only use unique data?
train_embedding = True      # train the embedding of numbers?
ninp = len(AB)              # dimension of RNN input

nseq = 5000
ntest = 500
pad = 0

# optimisation parameters
nepoch = 2000               # max how many times to go over data
alg = optim.Adam
dlargs = {'num_workers': 2, 
          'batch_size': 64, 
          'shuffle': True}  # dataloader arguments
optargs = {'lr': 5e-4}      # optimiser args
criterion = torch.nn.NLLLoss()

if forget_switch is not None:
    ninp += 1
    AB_ = AB+[forget_switch] # extended alphabet
else:
    AB_ = AB

# train and test
accuracy = np.zeros((4,9))
for i, Ls in enumerate([[5],[2,5],[2,5,7],[2,4,5,7]]):
    print('TRAINING %s NETWORK' % str(Ls))
    
    # housekeeping
#    nseq = int(np.min([nseq_max, (1-test_prop)*unique_seqs(AB, L)]))
#    ntest = int(np.ceil(nseq*test_prop))
#    lenseq = 2*L-1 
#    if forget_switch is not None:
#        lenseq += 1
#    
#    # draw data & targets
#    nums, ans = draw_seqs(L, nseq+ntest, Om=AB, switch=forget_switch)
#    trains = np.random.choice(nseq+ntest, nseq, replace=False)
#    tests = np.setdiff1d(np.arange(nseq+ntest), trains)
#    
#    if train_embedding:
#        enc = nums[trains,:] # (lenseq, nseq, ninp)
#        ptnums = torch.tensor(enc).type(torch.LongTensor)
#    else:
#        enc = as_indicator(nums[trains,:], AB_) # (lenseq, nseq, ninp)
#        ptnums = torch.tensor(enc).type(torch.FloatTensor) # convert to pytorch
#    ptans = torch.tensor(ans[trains]).type(torch.FloatTensor)
    
    nums, ans, numstest, anstest = make_dset(Ls, AB, int(nseq/len(Ls)), 
                                             ntest, 
                                             forget_switch,
                                             padding=pad)
    
    ptnums = torch.tensor(nums).type(torch.LongTensor)
    ptans = torch.tensor(ans).type(torch.FloatTensor)
    
    # specify model and loss
    rnn = RNNModel(rnn_type, len(AB_), ninp, nneur, nlayers, embed = train_embedding)
    
    loss_ = rnn.train(ptnums, ptans, optargs, dlargs, algo=alg, 
                      nepoch=nepoch, criterion=criterion,
                      do_print=False, epsilon=5e-4, padding=pad)
    
    print('TESTING %s NETWORK' % str(Ls))
    for j, l in enumerate(range(2,11)):
        print('On L = %d' %(l))
#        if l in Ls:
#            test_nums = nums[tests,:]
#            test_ans = ans[tests]
#            test_nums = numstest[l]
#            test_ans = anstest[l]
#        else:
        test_nums, test_ans = draw_seqs(l, ntest, Om=AB,
                                        switch=forget_switch)
        n_test, lseq = test_nums.shape
        
        O = torch.zeros(lseq, l, n_test)
        for inp in range(n_test):
            
            tst = test_nums[inp,:]
            
            if train_embedding:
                enc = tst
                test_inp = torch.tensor(enc).type(torch.LongTensor)
            else:
                enc = as_indicator(tst[np.newaxis,:], AB_) #(lenseq, nseq, ninp)
                test_inp = torch.tensor(enc).type(torch.FloatTensor).transpose(1,0)
            
            hid = rnn.init_hidden(1)
            for t in range(lseq):
                out, hid = rnn(test_inp[t:t+1,...], hid)
                O[t,:,inp] = torch.exp(out[0,0,tst[:l]])
                
        O = O.detach().numpy()
        
        whichone = np.argmax(O[-1,:,:],axis=0) # index in sequence of unique number
        accuracy[i,j] = np.mean(test_nums[np.arange(n_test), whichone] == test_ans)
    print('- '*20)

print('done')    
print(':' + ')'*12)

plt.plot(np.arange(2,11),accuracy.T)
plt.plot(np.arange(2,11),1/np.arange(2,11),'k--')
plt.title('GRU implicit generalisation, trained encoder')
plt.ylabel('Accuracy on test data')
plt.xlabel('L of test data')
plt.legend(['2','4','5','7','chance'])

#%% load from habanero

fracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,3.0]
Ps = [10,100,1000]
L = 3
rnn_type = 'GRU'

accuracy = np.zeros((len(Ps),len(fracs),9))*np.nan
loss = np.zeros((len(Ps),len(fracs),200000))*np.nan

for i,P in enumerate(Ps):
    for j,f in enumerate(fracs):
        N = int(f*P)
        
        try:
            params_fname = 'parameters_%d_%d_%d_%s.pt'%(N, P, L, rnn_type)
            loss_fname = 'loss_%d_%d_%d_%s.npy'%(N, P, L, rnn_type)
            accy_fname = 'accuracy_%d_%d_%d_%s.npy'%(N, P, L, rnn_type)
                
            acc = np.load(CODE_DIR+'results/'+accy_fname)
            los = np.load(CODE_DIR+'results/'+loss_fname)
            idx = np.min([200000, los.shape[0]])
            
            accuracy[i,j,:] = acc
            loss[i,j,:idx] = los[:idx]
        except:
            print('whoops: %d %f' % (P,f))


#%% test and look inside
special_nums, spc_ans = draw_seqs(L, nseq, Om=AB, switch=forget_switch, mirror=True)
n_test = special_nums.shape[0]

H = torch.zeros(lenseq, rnn.nhid, n_test)
O = torch.zeros(lenseq, L, n_test)
for inp in range(n_test):
    
    test_num = special_nums[inp,:]
    
    enc = as_indicator(test_num[np.newaxis,:], AB_).transpose(1,0,2) #(lenseq, nseq, ninp
    test_inp = torch.tensor(enc).type(torch.FloatTensor)
    
    hid = rnn.init_hidden(1)
    for t in range(lenseq):
        out, hid = rnn(test_inp[t:t+1,:,:], hid)
        O[t,:,inp] = torch.exp(out[0,0,test_num[:L]])
        H[t,:,inp] = hid.squeeze()

O = O.detach().numpy()
H = H.detach().numpy()

whichone = np.argmax(O[-1,:,:],axis=0) # index in sequence of unique number
accrcy = np.mean(special_nums[np.arange(n_test), whichone] == spc_ans) # fraction correct
print(accrcy)

#%% train decoders
SMdecoder = stateDecoder(AB_, nneur)
#tensH = torch.tensor(H.transpose(0,2,1))
#tensLab = torch.tensor(special_nums)

nepoch = 50 # how many times to go over data
alg = optim.Adam
dlargs = {'num_workers': 2, # dataloader arguments
          'batch_size': 4,  # keep in mind this is scaled by lenseq
          'shuffle': True}
optargs = {'lr': 1e-3}

LOSS = SMdecoder.train(tensH, tensLab, optargs,
                       dlargs, algo = alg, nepoch=nepoch)

#%% plot
cmap_name = 'spring'
cols = getattr(cm, cmap_name)(np.linspace(0,1,len(AB_)))
    
for tk, tok in enumerate(AB_):
    if (tok is forget_switch) or not (tok in test_num):
        continue
    plt.plot(is_memory_active(test_num, tok), '--', linewidth=1.5, c=cols[tk,:])
    plt.plot(np.exp(O[:,tk]), c=cols[tk,:])
    
plt.plot([L,L],plt.ylim(), 'k-')



