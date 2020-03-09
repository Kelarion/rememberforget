CODE_DIR = r'/home/matteo/Documents/github/rememberforget/'
svdir = r'~/Documents/uni/columbia/sueyeon/experiments/'

import sys, os
sys.path.append(CODE_DIR)

from sklearn import svm, calibration, linear_model, discriminant_analysis, manifold
import scipy.stats as sts
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.linalg as la
import umap
from cycler import cycler

from students import RNNModel, LinearDecoder

from needful_functions import *

#%% Load from Habanero experiment
#fracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,3.0]
fracs = [1,2,3,4,5,6,7,8,9,10,15,20,30]
Ps = [10,15,20,25,30]
#dts = [None, 10, 15, 20, 30, 40, 50, 100]
dts = [None]
FORGET = False
Q = None
rnn_type = 'tanh'
explicit = True and FORGET
persistent = True and FORGET
embed = False
dead_time = None
smoothening = 1 # of the loss function

FOLDERS = expfolders(explicit, persistent, embed, FORGET)

#if Q is not None:
#    signal_tokens=list(range(Q))
#else:
signal_tokens=None

accuracy = np.zeros((len(Ps),len(fracs),max(Ps)+1,len(dts)))*np.nan
trained_Ls = [[] for _ in Ps]
loss = np.zeros((len(Ps),len(fracs),200000))

rnns = [[[] for _ in fracs] for _ in Ps]
for i,P in enumerate(Ps):
    test_L = list(range(2,P+1))
    
    for j,N in enumerate(fracs):
        
#        wa = int(P*dead_time) if dead_time is not None else None
        wa = dead_time
        expinf = suffix_maker(N, P, rnn_type, Q, wa)
        params_fname = 'parameters'+expinf+'.pt'
        loss_fname = 'loss'+expinf+'.npy'
        args_fname = 'arguments'+expinf+'.npy'
        accy_fname = 'accuracy'+expinf+'.npy'
        dset_fname = 'datasets'+expinf+'.pkl'
        
        if not os.path.exists(CODE_DIR+FOLDERS+loss_fname):
            print(CODE_DIR+FOLDERS+loss_fname+' doesn`t exist!')
            continue
        
        ninp = P+1*(explicit or persistent)
        rnn = RNNModel(rnn_type, ninp, ninp, N, 1, embed=embed, 
                       persistent=persistent)
        rnn.load(CODE_DIR+FOLDERS+params_fname)
        rnns[i][j] = rnn
        
        for k, _dt in enumerate(dts):
            if os.path.exists(CODE_DIR+FOLDERS+args_fname):
                dsargs = np.load(CODE_DIR+FOLDERS+args_fname, allow_pickle=True).item()
                trained_Ls[i] = dsargs['Ls']
                if type(dsargs['Ls']) is not list:
                    dsargs['Ls'] = [dsargs['Ls']]
                these_L = [l for l in test_L if l not in dsargs['Ls']] # setdiff
    #            these_L = test_L
                if len(these_L) != 0:
                    acc1 = test_net(rnn, list(range(P)), these_L, 500, dead_time=_dt, 
                                    be_picky=None, signal_tokens=signal_tokens,
                                    explicit=explicit)[0]
                    accuracy[i,j,these_L, k] = acc1
                    
                acc2 = np.load(CODE_DIR+FOLDERS+accy_fname)
                accuracy[i,j,dsargs['Ls'], k] = acc2
                
            else:
                acc1 = test_net(rnn, list(range(P)), test_L, 500, dead_time=_dt,
                                be_picky=None, signal_tokens=signal_tokens,
                                explicit=explicit)[0]
                accuracy[i,j,test_L, k] = acc1
        
#        Ls[test_L]
        
        print('done testing N=%d, P=%d network'%(N,P))
        
        los = np.load(CODE_DIR+FOLDERS+loss_fname)
        idx = np.min([200000, los.shape[0]])
        
        loss[i,j,:idx] = los[:idx]

loss_smooth = np.apply_along_axis(np.convolve,-1,loss,np.ones(smoothening),'same')

#%% 
pid = -1
cmap_name = 'copper'
cols = getattr(cm, cmap_name)(np.linspace(0,1,len(fracs)))
mpl.rcParams['axes.prop_cycle'] = cycler(color=cols)
plt.plot(loss_smooth[pid,:,:].T)
#plt.plot()
plt.legend(np.array(fracs), loc='upper right')
plt.ylabel('NLL loss')
plt.title('Training with P='+str(Ps[pid]))

#%%
pid = -4
t = 0
#plt.figure()
#plt.plot(test_L, accuracy[pid,:,:].T)
#plt.legend(fracs)
#plt.plot(test_L, 1/np.array(test_L), 'k--')
#plt.xlabel('test L')
#plt.ylabel('test accuracy')
plt.figure()
plt.imshow(accuracy[pid,:,:,t], 'binary')
plt.yticks(np.arange(accuracy.shape[1]), fracs)
plt.xticks(np.arange(accuracy.shape[2]))
#plt.ylim([-len(fracs)-0.5,0.5])
#plt.gca().set_yticklabels(fracs)
plt.gca().autoscale(enable=True)

plt.gca().get_images()[0].set_clim([0,1])
plt.colorbar()
plt.title('performance P=%d, trained on L=%s'%(Ps[pid], trained_Ls[pid]))
plt.xlabel('Sequence length (L)')
plt.ylabel('Number of neurons (N)')

#%% Linear decoding of memory activation
pind = Ps.index(10)
L = [4]
whichrnn = -1
P = Ps[pind]
_dt = None
decode_token = False
decode_from = 'hidden'
#decode_from = 'reset'
#decode_from = 'update'
#decode_time = 'input'
#decode_time = 'inter'
decode_time = 'all'

isgru = rnn_type in ['GRU', 'tanh-GRU']
if isgru:
    suff =suffix_maker(fracs[whichrnn], P, 'GRU', Q, wa)
    ninp = P+1*(explicit or persistent)
    rnn = RNNModel('tanh-GRU', ninp, ninp, fracs[whichrnn], 1, embed=embed,
                   persistent=persistent)
    rnn.load(CODE_DIR+FOLDERS+'parameters'+suff+'.pt')
else:
    rnn = rnns[pind][whichrnn]

nseq = 1500
clsfr = svm.LinearSVC
cfargs = {'tol': 1e-5, 'max_iter':5000}
#clsfr = discriminant_analysis.LinearDiscriminantAnalysis
#cfargs = {}

##### generate training data ##################################################
if isgru:
    acc, H, I, Z, R = test_net(rnn, list(range(P)), L, nseq, dead_time=_dt, 
                               be_picky=None, signal_tokens=signal_tokens,
                               explicit=explicit,
                               return_hidden=True, give_gates=True)
    Z = Z.squeeze() # lenseq x nneur x nseq
    R = R.squeeze() # lenseq x nneur x nseq
else: 
    acc, H, I = test_net(rnn, list(range(P)), L, nseq, dead_time=_dt,
                         be_picky=None, signal_tokens=signal_tokens,
                         explicit=explicit, return_hidden=True)
lseq = H.shape[0]
H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq

if decode_time == 'input':
    these_t = (I!=rnn.padding)
elif decode_time == 'inter':
    these_t = (I==rnn.padding)
elif decode_time == 'all':
    these_t = (I<=1e10)

mem_act = np.zeros(I.shape + (P+1,)) # lenseq x nseq x ntoken
tok_pres = np.zeros(I.shape + (P+1,)) # lenseq x nseq x ntoken

idx = np.nonzero((I.T!=-1)*(I.T!=P))[1].reshape((nseq,-1))
nrep = np.diff(idx, axis=1, append=lseq)
tok = np.take_along_axis(I,np.repeat(idx.flatten(),nrep.flatten()).reshape((nseq,-1)).T,0)
for p in range(P+1):
    mem_act[:,:,p] =np.apply_along_axis(is_memory_active, 0, I, p)
    tok_pres[:,:,p] = (tok==p)
    
#t_ = (np.cumsum(np.ones((lseq, nseq)), axis=0)-1).astype(int)
t_ = np.cumsum(I!=rnn.padding, axis=0)-1
#t_[t_<L[0]] = 0
#t_[t_>=L[0]] = 1
#t_ = np.zeros((lseq, nseq), dtype=int)

# make it nice
idx = np.nonzero(these_t.T)[1].reshape((nseq,-1)).T
if decode_from == 'hidden':
    X = np.take_along_axis(H, idx[:,None,:], 0)
elif decode_from == 'reset':
    X = np.take_along_axis(R, idx[:,None,:], 0)
elif decode_from == 'update':
    X = np.take_along_axis(Z, idx[:,None,:], 0)
t_ = np.take_along_axis(t_, idx, 0)

if decode_token:
    ans = np.take_along_axis(tok_pres, idx[:,:,None], 0)
else:
    ans = np.take_along_axis(mem_act, idx[:,:,None], 0)
ans[:,:,-1] = t_ >= L[0]

##### fit models ##############################################################
clf = LinearDecoder(rnn, P, clsfr)
clf.fit(X, ans, t_, **cfargs)

coefs = clf.coefs
thrs = clf.thrs

##### generate test data ######################################################
if isgru:
    acc, H, I, Z, R = test_net(rnn, list(range(P)), L, nseq, dead_time=_dt, 
                               be_picky=None, signal_tokens=signal_tokens,
                               explicit=explicit,
                               return_hidden=True, give_gates=True)
    Z = Z.squeeze() # lenseq x nneur x nseq
    R = R.squeeze() # lenseq x nneur x nseq
else: 
    acc, H, I = test_net(rnn, list(range(P)), L, nseq, dead_time=_dt,
                         be_picky=None, signal_tokens=signal_tokens,
                         explicit=explicit, return_hidden=True)
H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq

if decode_time == 'input':
    these_t = (I!=rnn.padding)
elif decode_time == 'inter':
    these_t = (I==rnn.padding)
elif decode_time == 'all':
    these_t = (I<=1e10)

mem_act = np.zeros(I.shape + (P+1,)) # lenseq x nseq x ntoken
tok_pres = np.zeros(I.shape + (P+1,)) # lenseq x nseq x ntoken

idx = np.nonzero((I.T!=-1)*(I.T!=P))[1].reshape((nseq,-1))
nrep = np.diff(idx, axis=1, append=lseq)
tok = np.take_along_axis(I,np.repeat(idx.flatten(),nrep.flatten()).reshape((nseq,-1)).T,0)
for p in range(P+1):
    mem_act[:,:,p] =np.apply_along_axis(is_memory_active, 0, I, p)
    tok_pres[:,:,p] = (tok==p)

t_proj = np.cumsum(I!=rnn.padding, axis=0)-1
#t_proj = np.ones(t_.shape, dtype=int)*1
#t_proj = (np.cumsum(I!=rnn.padding, axis=0)-1)
#t_proj[t_proj<L[0]] = 0
#t_proj[t_proj>=L[0]] = 1

idx = np.nonzero(these_t.T)[1].reshape((nseq,-1)).T
if decode_from == 'hidden':
    X = np.take_along_axis(H, idx[:,None,:], 0)
elif decode_from == 'reset':
    X = np.take_along_axis(R, idx[:,None,:], 0)
elif decode_from == 'update':
    X = np.take_along_axis(Z, idx[:,None,:], 0)
t_proj = np.take_along_axis(t_proj, idx, 0)

if decode_token:
    ans = np.take_along_axis(tok_pres, idx[:,:,None], 0)
else:
    ans = np.take_along_axis(mem_act, idx[:,:,None], 0)
ans[:,:,-1] = t_proj >= L[0]

### test peformance #########################################################
perf = clf.test(X, ans, t_proj)[:-1,:]
marg = clf.margin(X, ans, t_proj)

#pp = ans[:,:,:-1].mean() # i'm setting 'chance' level as the best you could do
#pmin = np.max([pp, 1-pp])  # with a constant classifier (using empirical )
pmin = 1-(1/L[0])

plt.figure()
plt.imshow(perf)

plt.title('decoder accuracies (Network performance: %.3f)'%acc[0])
plt.xlabel('time bin')
plt.ylabel('token')
plt.gca().get_images()[0].set_clim([pmin,1])
plt.colorbar()

plt.figure()
plt.imshow(marg,'coolwarm')

plt.title('decoder margins (Network performance: %.3f)'%acc[0])
plt.xlabel('time bin')
plt.ylabel('token')
pp = np.max(np.abs([marg.min(), marg.max()]))
plt.gca().get_images()[0].set_clim([-pp,pp])
plt.colorbar()

proj_ctr = clf.project(X, t_proj)
inner = np.einsum('ik...,jk...->ij...', coefs, coefs)

#%% Code for LDA
# get the projections onto LDs, and angle between different LDs
#coefs = np.zeros((P+1, H.shape[1], len(t_lbs)))
#thrs = np.zeros((P+1,len(t_lbs)))
#for p in range(P+1):
#    for t in t_lbs:
#        coefs[p,:,t] = clf[p][t].coef_/la.norm(clf[p][t].coef_)
#        thrs[p,t] = -clf[p][t].intercept_/la.norm(clf[p][t].coef_)

#inner = np.einsum('ik...,jk...->ij...', coefs, coefs) # dot product between LDs
#proj = np.einsum('ikj...,jk...->ij...', coefs, H) # projection onto LDs
#
## center the projections around the decision boundary
#proj_ctr = proj - thrs[:,:,np.newaxis]

#%% see how well the LDs separate token activations
#plot_these = list(range(lseq))
#plot_these = list(range(1,lseq,10))
plot_these = clf.time_bins
#p_fixed = 1

fig, axs = plt.subplots(P, len(plot_these))
for p in range(P):
    axs[p][0].set_ylabel('Token %d'%p)
    for t,tt in enumerate(plot_these):
        if p==0:
            axs[p,t].set_title('Time %d'%tt)
        
        x = proj_ctr[p, t_proj==tt]
        y = ans.transpose((2,0,1))
        ison = x[y[p,t_proj==tt]==1]
        isoff = x[y[p,t_proj==tt]==0]
#        pval = sts.ks_2samp(ison,isoff).pvalue
        
        axs[p,t].hist(ison, density=True, alpha=0.5)
        axs[p,t].hist(isoff, density=True, alpha=0.5)
        axs[p,t].plot([0,0],axs[p,t].get_ylim(), 'k-')
#        axs[p,t].text(axs[p,t].get_xlim()[0],axs[p,t].get_ylim()[1],
#                 'p={:.1e} (K-S)'.format(pval), fontsize=10, va='top')
        
        if p==P-1:
            axs[p][t].set_xlabel('LD1')

plt.figure()
plt.hist(proj_ctr[-1,t_proj<L[0]], alpha=0.5)
plt.hist(proj_ctr[-1,t_proj>=L[0]], alpha=0.5)
plt.legend(['remember','forget'])
plt.xlabel('projection onto first/second half classifier')

#%%
#foo = la.norm(coefs - coefs[:,:,:1], axis=1)
foo = la.norm(np.diff(coefs, axis=-1), axis=1)
plt.plot(clf.time_bins[1:],foo.T)
plt.legend(list(range(P+1)))
plt.ylabel('Distance from previous timestep')
plt.xlabel('time bin')

plt.figure() 
plt.plot(clf.time_bins, thrs.T)
plt.legend(list(range(P+1)))
plt.ylabel('Classifier decision boundary')
plt.xlabel('time')

#%% see how different LDs relate to each other
#plot_these = list(range(lseq))
plot_these = clf.time_bins
#plot_these = [1, L[0], -1]

fig, axs = plt.subplots(1, len(plot_these))
for t,tt in enumerate(plot_these):
    axs[t].imshow(inner[:,:,tt], cmap='bwr')
    axs[t].set_title('Time %d'%tt)
    axs[t].set_xlabel('Token')
    axs[t].get_images()[0].set_clim([-1,1])
    if t != 0:
        axs[t].set_yticks([])

axs[0].set_ylabel('Token')
#fig.colorbar()

fig, axs = plt.subplots(1, len(plot_these))
for t,tt in enumerate(plot_these):
    axs[t].imshow(coefs[:,:,tt].T, cmap='bwr')
    axs[t].set_title('Time %d'%tt)
    axs[t].set_xlabel('Token')
    axs[t].get_images()[0].set_clim([-1,1])
    if t != 0:
        axs[t].set_yticks([])

axs[0].set_ylabel('Neuron')

#%% how do the projections evolve for example sequences?
mpl.rcParams['axes.prop_cycle'] = cycler(color='rgbcymk')
cmap_name = 'jet'
which_seq = 3

these_toks = np.isin(list(range(P+1)), I[:,which_seq])
labs = I[:,which_seq].astype(int)
labpos = labs.argsort()[-np.sum(labs!=rnn.padding):]
#half = (labs==P).argmax()

cols = getattr(cm, cmap_name)(np.linspace(0,1,sum(these_toks)))

plt.axes()
plt.gca().set_prop_cycle(cycler(color=cols))
plt.plot(proj_ctr[these_toks,:,which_seq].T)
plt.xticks(ticks=labpos,labels=labs[labpos])
plt.legend(np.arange(P+1)[these_toks])
plt.plot(plt.xlim(),[0,0], 'k--')
#plt.plot([half,half],plt.ylim(), 'k-')

c = cols[np.unique(labs[labpos], return_inverse=True)[1], :]
plt.gca().set_prop_cycle(cycler(color=c))
plt.plot(np.repeat(labpos[:,None],2,axis=1).T, plt.ylim(), '--')

plt.title('Projections onto best LD of each token & time')
plt.ylabel('Projection (centred at decision boundary)')
plt.xlabel('Token at time t')

if isgru:
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    ax1.imshow(Z[:,:,which_seq].T, cmap='binary')
    ax1.get_images()[0].set_clim([0,1])
    ax1.set_xticks(labpos)
    ax1.set_xticklabels(labs[labpos])
    ax1.plot(np.repeat(labpos[:,None],2,axis=1).T, ax1.get_ylim(), 'k--', alpha=0.5)
    ax1.set_title('Update gate activities')
    
    ax2.imshow(R[:,:,which_seq].T, cmap='binary')
    ax2.get_images()[0].set_clim([0,1])
    ax2.set_xticks(labpos)
    ax2.set_xticklabels(labs[labpos])
    ax2.plot(np.repeat(labpos[:,None],2,axis=1).T, ax2.get_ylim(), 'k--', alpha=0.5)
    ax2.set_title('Reset gate activities')
    plt.colorbar(ax2.get_images()[0])

plt.figure()
plt.imshow(H[:,:,which_seq].T)
#plt.plot(H[:,:,which_seq])
plt.xticks(ticks=labpos,labels=labs[labpos])
plt.plot(np.repeat(labpos[:,None],2,axis=1).T, plt.ylim(), 'k--', alpha=0.5)
plt.title('Neural activations in example sequence')
plt.xlabel('Token presented at time t')
plt.ylabel('Neuron')

#%% Decode time in trial
pind = Ps.index(10)
L = [7]
whichrnn = -2
P = Ps[pind]

isgru = rnn_type in ['GRU', 'tanh-GRU']
if isgru:
    suffix_maker(fracs[whichrnn], P, 'GRU', Q)
    rnn = RNNModel('tanh-GRU', ninp, ninp, fracs[whichrnn], 1, embed=embed,
                   persistent=persistent)
    rnn.load(CODE_DIR+FOLDERS+'parameters'+suffix_maker(fracs[whichrnn], P, 'GRU', Q)+'.pt')
else:
    rnn = rnns[pind][whichrnn]

nseq = 1500
#clsfr = calibration.CalibratedClassifierCV(svm.LinearSVC(tol=1e-5, max_iter=5000), cv=10)
clsfr = svm.LinearSVC
#clsfr = linear_model.LogisticRegression(tol=1e-5, solver='lbfgs', max_iter=1000)
#clsfr = discriminant_analysis.LinearDiscriminantAnalysis
cfargs = {'tol': 1e-5, 'max_iter':5000}

# generate training data
acc, H, I = test_net(rnn, list(range(P)), L, nseq, dead_time=dead_time,
                     be_picky=True, signal_tokens=signal_tokens,
                     explicit=explicit, return_hidden=True)
lseq = H.shape[0]

#H = H.transpose(1,0,2,3).reshape((H.shape[1],-1))
H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq
H_flat = H.transpose(1,0,2).reshape((H.shape[1],-1))

ans = np.tile(np.arange(lseq)[:,np.newaxis],(1,H.shape[2]))
ans_flat = ans.flatten()

# do the decoding
clf = clsfr(**cfargs)
clf.fit(H_flat.T, ans_flat)

# test
if isgru:
    acc, H, I, Z, R = test_net(rnn, list(range(P)), L, nseq, dead_time=dead_time, 
                               be_picky=True, signal_tokens=signal_tokens,
                               explicit=explicit,
                               return_hidden=True, give_gates=True)
    Z = Z.squeeze() # lenseq x nneur x nseq
    R = R.squeeze() # lenseq x nneur x nseq
else: 
    acc, H, I = test_net(rnn, list(range(P)), L, nseq, dead_time=dead_time,
                         be_picky=True, signal_tokens=signal_tokens,
                         explicit=explicit, return_hidden=True)

H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq
H_flat = H.transpose(1,0,2).reshape((H.shape[1],-1))

plt.figure()
plt.plot(clf.predict(H_flat.T))
plt.plot(ans_flat, 'k--')
plt.legend(['predicted time','actual time'])
plt.xlabel('time point, ordered by position in trial')
plt.ylabel('position in trial')
plt.title('Accuracy: %.3f'%clf.score(H_flat.T, ans_flat))

#%%
C = clf.coef_/la.norm(clf.coef_, axis=-1)[:, np.newaxis]
inner = C@C.T
proj = C@H_flat
intcpt = clf.intercept_/la.norm(clf.coef_, axis=-1)

plt.figure()
plt.imshow(C.T, cmap='bwr')
plt.gca().get_images()[0].set_clim([-1,1])
plt.xlabel('time point')
plt.ylabel('neuron')
plt.title('Classifier weights')
plt.colorbar()

# center the projections around the decision boundary
fig, axs = plt.subplots(2,int(lseq/2))
for t in range(lseq):
    r = round((t+1)/lseq)
    c = t%int(lseq/2)
    axs[r,c].hist(proj[t, ans_flat==t], density=True, alpha=0.5, color='r')
    for tt in range(lseq):
        if tt != t:
            axs[r,c].hist(proj[t, ans_flat == tt], density=True, alpha=0.5)
    axs[r,c].set_title('Time %d'%t)
    axs[r,c].legend(np.concatenate(([t], np.setdiff1d(range(lseq),t))))
    axs[r,c].plot([intcpt[t], intcpt[t]], axs[r,c].get_ylim(), 'k-')
    if c!=0:
        axs[r,c].set_yticks([])
    if r!=1:
        axs[r,c].set_xticks([])
    axs[r,c].set_xlabel('Proj onto t=%d  weights'%t)


## <whinging>
## I'm a real big fan of how there is a set of methods in mpl.pyplot, and the 
## TOTALLY EQUIVALENT methods in mpl.axes.Axes mostly have DIFFERENT NAMES
## it makes my life SO MUCH EASIER
##
## And a side-note on python syntax,
## To concatenate vector 'a' to matrix 'B' in:
## numpy: np.concatenate((a[np.newaxis, :], B))
## matlab: [a, B] 
## </whinging>

#%% Looking at projections of the activity
@numba.njit()
def test_dist(a,b):
    return np.abs(np.sum(a*b))
    
#%%
X = H.transpose(1,0,2).reshape((H.shape[1],-1)) # hidden
#X = Z.transpose(1,0,2).reshape((Z.shape[1],-1)) # update gate
#X = R.transpose(1,0,2).reshape((R.shape[1],-1)) # reset gate

tim = np.tile(np.arange(lseq)[:,np.newaxis],(1,H.shape[2]))
tim = tim.flatten()
trial = np.tile(np.arange(nseq)[:,np.newaxis],(1,H.shape[0]))
trial = trial.T.flatten() 
token = I.flatten()
firsthalf = t_proj.flatten() >= L[0]
whichmemory = ans[:,:,0].flatten() + 2*ans[:,:,1].flatten()

# do UMAP
#reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.8)
#emb = reducer.fit_transform(X.T)

# do PCA
_, S, V = la.svd(X[:,:].T-X[:,:].mean(1), full_matrices=False)
pcs = X.T@V[:3,:].T

plt.figure()
plt.loglog((S**2)/np.sum(S**2))
plt.xlabel('PC')
plt.ylabel('variance explained')

# do MDS
#print('Fitting MDS')
#mds1 = manifold.MDS(n_components=3, eps=1e-4)
#mdsemb_first = mds1.fit_transform(X[:,firsthalf].T)
#
#mds2 = manifold.MDS(n_components=3, eps=1e-4)
#mdsemb_second = mds2.fit_transform(X[:,~firsthalf].T)

#%% PCA
#plotthese = whichmemory > 0
plotthese = token!=-1
#cmap_name = 'nipy_spectral'
cmap_name = 'bwr'
#colorby = token[plotthese]
#colorby = tim
colorby = firsthalf[plotthese]
#colorby = whichmemory[plotthese]

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter(pcs[plotthese,0],pcs[plotthese,1],pcs[plotthese,2], c=colorby, alpha=0.1, cmap=cmap_name)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.draw_all()

#plt.box(False)

#%% UMAP
cmap_name = 'nipy_spectral'
#colorby = token[token!=-1]
colorby = tim
#colorby = t_proj.flatten()
#colorby = firsthalf[token!=-1]
#colorby = whichmemory[token!=-1]

# plot
plt.figure()
scat = plt.scatter(emb[:,0],emb[:,1], c=colorby, alpha=0.1, cmap=cmap_name)

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.draw_all()
#cb.set_label('Time in trial')

plt.xlabel('umap dimension 1')
plt.ylabel('umap dimension 2')

#%%
tok = 8
tr = 13

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(emb[:,0],emb[:,1], c=[[0.5,0.5,0.5]], alpha=0.01)
ax.scatter(emb[trial==tr, 0],emb[trial==tr, 1], c='b')

for t, i in enumerate(zip(I[:,0], emb[trial==tr, :])):
    ax.annotate('t=%d: %d'%(t,i[0]), i[1])

ax.legend(['not  trial', 'trial'])

plt.xlabel('umap dimension 1')
plt.ylabel('umap dimension 2')

#ax.scatter(emb[firsthalf & istok_on,0], 
#           emb[firsthalf & istok_on,1], 
#           emb[firsthalf & istok_on,2])
#ax.scatter(emb[firsthalf & istok_off,0], 
#           emb[firsthalf & istok_off,1], 
#           emb[firsthalf & istok_off,2])
#
#ax.scatter(emb[~firsthalf & istok_on,0], 
#           emb[~firsthalf & istok_on,1],
#           emb[~firsthalf & istok_on,2])
#ax.scatter(emb[~firsthalf & istok_off,0], 
#           emb[~firsthalf & istok_off,1],
#           emb[~firsthalf & istok_off,2])

#plt.scatter(emb[firsthalf & istok_on,0],emb[firsthalf & istok_on,1])
#plt.scatter(emb[firsthalf & istok_off,0],emb[firsthalf & istok_off,1])
#plt.scatter(emb[~firsthalf & istok_on,0],emb[~firsthalf & istok_on,1])
#plt.scatter(emb[~firsthalf & istok_off,0],emb[~firsthalf & istok_off,1])

#ax.scatter(emb[:,0],emb[:,1], c=[[0.5,0.5,0.5]], alpha=0.01)




