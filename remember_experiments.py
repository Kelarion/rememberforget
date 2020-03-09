CODE_DIR = r'/home/matteo/Documents/github/rememberforget/'
svdir = r'~/Documents/uni/columbia/sueyeon/experiments/'

import sys, os, re
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

from students import RNNModel, LinearDecoder, Dichotomies

from needful_functions import *

#%% Load from Habanero experiment
#N_list_overwrite = [10]
N_list_overwrite = [3,5,7,10,15]
Q_list_overwrite = None
# Q_list_overwrite = [2,3,4,5]
P = 30
L = '15' # length of training sequences
rnn_type = 'tanh'
embed = False
smoothening = 1 # of the loss function

FOLDERS = expfolders(False, False, embed, False)

exp_params, results = batch_test(CODE_DIR+FOLDERS, 
                                 rnn_type, P, L,
                                 N_list=N_list_overwrite,
                                 Q_list=Q_list_overwrite)

N_list, Q_list, trained_Ls = exp_params
rnns, metrics, accuracy = results

#%% plot the loss
nid = 4
# show_me = 'train_loss'
show_me = 'test_loss'
# show_me = 'train_orthog'
# show_me = 'test_orthog'
# show_me = 'test_parallelism'
# show_me = 'train_parallelism'
cmap_name = 'spring'

cols = getattr(cm, cmap_name)(np.linspace(0,1,len(Q_list)))
mpl.rcParams['axes.prop_cycle'] = cycler(color=cols)
plt.plot(metrics[show_me][:,nid,:].T)
#plt.plot()
plt.legend(np.array(Q_list), loc='upper right')
plt.ylabel(show_me)
plt.title('Training with N='+str(N_list[nid]))

#%% plot the accuracy
nid = 2

acc_sig = np.array([np.nanmean(np.exp(-accuracy[i,nid,:,:q]),-1)\
                    for i,q in enumerate(Q_list)])
acc_noise = np.array([np.nanmean(np.exp(-accuracy[i,nid,:,q:]),-1)\
                      for i,q in enumerate(Q_list)])
plt.figure()
ba = plt.imshow(acc_sig,'binary')
#ba = plt.imshow(acc_noise,'binary')
ba.cmap.set_bad(color='red', alpha=0.5)
plt.yticks(np.arange(accuracy.shape[0]), Q_list)
plt.xticks(np.arange(2,accuracy.shape[2]))
plt.xlim([2,accuracy.shape[2]])
#plt.ylim([-len(fracs)-0.5,0.5])
#plt.gca().set_yticklabels(fracs)
plt.gca().autoscale(enable=True)

plt.gca().get_images()[0].set_clim([0,1])
cb = plt.colorbar()
cb.set_label('Mean exp(loss)')
plt.title('performance N=%d, trained on L=%s'%(N_list[nid], L))
plt.xlabel('Test sequence length (L)')
plt.ylabel('Number of signal tokens (Q)')

#%%
nid = 2
psid = 8
cmap_name = 'spring'
# show_me = 'train'
show_me = 'test'

# PS = np.nanmean(metrics[show_me+'_parallelism'][:,:,nid,:],axis=0)
PS = metrics[show_me+'_parallelism'][:,psid,nid,:]
loss = np.repeat(metrics[show_me+'_loss'][psid:psid+1,nid,:],10,axis=0)
tmax = (np.isnan(PS)|np.isnan(loss)).argmax(1)
PS = PS[:,:np.max(tmax)]
loss = loss[:,:np.max(tmax)] 

final_PS = PS[np.arange(PS.shape[0]),tmax-1]
final_loss = loss[np.arange(PS.shape[0]),tmax-1]

nq = (np.ones(PS.shape).T*np.arange(PS.shape[0])).T.flatten()
tim = np.cumsum(~np.isnan(PS),axis=1)
tim = (tim/tim.max(1)[:,None]).flatten()

# plt.scatter(loss.flatten(),PS.flatten(), c=nq, alpha=0.2, s=5) 
sct = plt.scatter(final_loss, final_PS, marker='*', s=200, edgecolor='k')

plt.legend(sct.legend_elements()[0],Q_list)
plt.ylim([0,1.1])

plt.xlabel(show_me+' loss')
plt.ylabel('parallelism')
plt.title('Network of %d units'%N_list[nid])

#%%
qind = 0
nind = 1
#Ls = [4]
decode_token = False
decode_from = 'hidden'
#decode_from = 'reset'
#decode_from = 'update'
#Q = Q_list[qind]
these_Q = Q_list
# these_L = [15]
these_L = list(range(2,30))

isgru = rnn_type in ['GRU', 'tanh-GRU']

nseq = 1500
clsfr = svm.LinearSVC
cfargs = {'tol': 1e-5, 'max_iter':5000}
#clsfr = discriminant_analysis.LinearDiscriminantAnalysis
#cfargs = {}

perf = np.zeros((len(these_Q), len(these_L)))
marg = np.zeros((len(these_Q), len(these_L)))
dimension = np.zeros(len(these_Q))
for i,q in enumerate(these_Q):
    
    if isgru:
        suff = suffix_maker(N_list[nind], P, L, rnn_type, q, None)
        rnn = RNNModel('tanh-GRU', P, P, N_list[nind], 1, embed=embed,
                       persistent=False)
        rnn.load(CODE_DIR+FOLDERS+'parameters'+suff+'.pt')
    else:
        rnn = rnns[i][nind]
    
    for j,l in enumerate(these_L):
    
        ##### generate training data ##########################################
        X, I, t_, ans, _ = draw_hidden_for_decoding(rnn, [l], nseq, 
                                                    chunking=None,
                                                    decode_from=decode_from, 
                                                    decode_token=decode_token,
                                                    be_picky=None, 
                                                    forget=False, 
                                                    dead_time=None)
        
        ##### fit models ######################################################
        clf = LinearDecoder(rnn, P, clsfr)
        clf.fit(X, ans, t_, **cfargs)
        
        coefs = clf.coefs
        thrs = clf.thrs
        
        # X_ = X.transpose(1,0,2).reshape((X.shape[1],-1))
        # _, S, V = la.svd(X_[:,:].T-X_[:,:].mean(1), full_matrices=False)
        # dimension[i] = (np.sum(S**2)**2)/np.sum(S**4)
        
        ##### generate test data ##############################################
        X, I, t_proj, ans, acc = draw_hidden_for_decoding(rnn, [l], nseq, 
                                                          chunking=None,
                                                          decode_from=decode_from, 
                                                          decode_token=decode_token,
                                                          be_picky=None, 
                                                          forget=False, 
                                                          dead_time=None)
        
        ### test peformance ###################################################
        perf[i,j] = np.mean(clf.test(X, ans, t_proj)[:q])
        marg[i,j] = np.mean(clf.margin(X, ans, t_proj)[:q])
        
    print('Done with Q=%d'%q)

#pp = ans[:,:,:-1].mean() # i'm setting 'chance' level as the best you could do
#pmin = np.max([pp, 1-pp])  # with a constant classifier (using empirical )
#pmin = 1-(1/L)
pmin = 0.75

plt.figure()
plt.imshow(perf)

#net_acc = np.exp(-acc[0][:Q]).mean()
#plt.title('decoder accuracies (Network performance: %.3f)'%net_acc)
plt.title('decoder accuracies') 
plt.yticks(np.arange(perf.shape[0]), Q_list)
plt.xticks(np.arange(P-2), np.arange(2,P))
plt.xlabel('Test sequence length (L)')
plt.ylabel('Number of signal tokens (Q)')
plt.gca().get_images()[0].set_clim([pmin,1])
plt.gca().autoscale(enable=True)
plt.colorbar()

plt.figure()
plt.imshow(marg,'coolwarm')

#plt.title('decoder margins (Network performance: %.3f)'%net_acc)
plt.title('decoder margins') 
plt.yticks(np.arange(perf.shape[0]), Q_list)
plt.xticks(np.arange(P-2), np.arange(2,P))
plt.xlabel('Test sequence length (L)')
plt.ylabel('Number of signal tokens (Q)')
# pp = np.max(np.abs([marg.min(), marg.max()]))
pp = marg.std()
plt.gca().get_images()[0].set_clim([-pp,pp])
plt.gca().autoscale(enable=True)
plt.colorbar()

proj_ctr = clf.project(X, t_proj)
inner = np.einsum('ik...,jk...->ij...', coefs, coefs)

#%% Plot the decoder weights in various ways
plt.imshow(inner[:,:,0], 'bwr')
plt.gca().get_images()[0].set_clim([-1,1])
plt.colorbar()

plt.figure()
plt.imshow(clf.coefs[:,:,0].T, 'bwr')
plt.gca().get_images()[0].set_clim([-1,1])
plt.colorbar()

plt.figure()
plt.scatter(clf.coefs[0,:,0],clf.coefs[1,:,0])
lims = [min([plt.xlim()[0],plt.ylim()[0]]), max([plt.xlim()[1],plt.ylim()[1]])]
plt.gca().set_xlim(lims)
plt.gca().set_ylim(lims)
plt.gca().set_aspect('equal')
plt.plot(lims, lims, 'k--', alpha=0.2)
plt.plot(lims,[0,0], 'k-.', alpha=0.2)
plt.plot([0,0],lims, 'k-.', alpha=0.2)
plt.xlabel('Token 0 weight')
plt.ylabel('Token 1 weight')

#%% Do the PCA plots
# make sure you've only done one Q above
lseq = I.shape[0]
q = these_Q[-1]
X_ = X.transpose(1,0,2).reshape((X.shape[1],-1))

tim = np.tile(np.arange(lseq)[:,np.newaxis],(1,X.shape[2]))
tim = tim.flatten()
trial = np.tile(np.arange(nseq)[:,np.newaxis],(1,X.shape[0]))
trial = trial.T.flatten() 
token = I.flatten()
#firsthalf = t_proj.flatten() >= L[0]
# whichmemory = ans[:,:,0].flatten() + 2*ans[:,:,1].flatten()
whichmemory = np.sum(ans[:,:,:q]*(2**np.arange(q)),axis=2).flatten()

# do UMAP
#reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.8)
#emb = reducer.fit_transform(X.T)

# do PCA
_, S, V = la.svd(X_[:,:].T-X_[:,:].mean(1), full_matrices=False)
pcs = X_.T@V[:3,:].T

plt.figure()
plt.loglog((S**2)/np.sum(S**2))
plt.xlabel('PC')
plt.ylabel('variance explained')

#%% PCA
#plotthese = whichmemory > 0
plotthese = token!=-1
cmap_name = 'nipy_spectral'
# cmap_name = 'viridis'
# colorby = token[plotthese]
#colorby = tim
#colorby = firsthalf[plotthese]
colorby = whichmemory[plotthese]

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

#%% coeff plots
plotthese = token!=-1
cmap_name = 'nipy_spectral'
# cmap_name = 'viridis'
#colorby = token[plotthese]
#colorby = tim
#colorby = firsthalf[plotthese]
colorby = whichmemory

p1 = proj_ctr[0,:,:].flatten()
p2 = proj_ctr[1,:,:].flatten()
# plot
fig = plt.figure()
scat = plt.scatter(p1, p2, c=colorby, alpha=0.8, cmap=cmap_name)

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.draw_all()
#cb.set_label('Time in trial')

plt.xlabel('Token 0 classifier')
plt.ylabel('Token 1 classifier')

#%%
# epoch = 0
l = 15
q = 5
P = 30
N = 5

nseq = 1500
decode_token = False
decode_from = 'hidden'

clsfr = svm.LinearSVC
cfargs = {'tol': 1e-5, 'max_iter':5000}

perf = np.zeros(578)
marg = np.zeros(578)
signal_loss = np.zeros(578)
noise_loss = np.zeros(578)
dimension = np.zeros(578)
shat_dim = np.zeros((578,3))
for epoch in range(578):
    rnn = RNNModel('tanh', P, P, N, 1, embed=False, persistent=False)
    rnn.load('/home/matteo/Documents/github/rememberforget/results/justremember' + 'params_epoch%d.pt'%epoch)
    
    ##### generate training data ##########################################
    X, I, t_, ans, _ = draw_hidden_for_decoding(rnn, [l], nseq, 
                                                chunking=None,
                                                decode_from=decode_from, 
                                                decode_token=decode_token,
                                                be_picky=None, 
                                                forget=False, 
                                                dead_time=None)
    
    ##### fit models ######################################################
    clf = LinearDecoder(rnn, P, clsfr)
    clf.fit(X, ans, t_, **cfargs)
    
    coefs = clf.coefs
    thrs = clf.thrs
    
    X_ = X.transpose(1,0,2).reshape((X.shape[1],-1))
    _, S, V = la.svd(X_[:,:].T-X_[:,:].mean(1), full_matrices=False)
    dimension[epoch] = (np.sum(S**2)**2)/np.sum(S**4)
    
    dclf = LinearDecoder(rnn, 3, clsfr)
    # for d in Dichotomies(ans[:,:,:2], 'general'):
    dclf.fit(X, np.array([d for d in  Dichotomies(ans[:,:,:2], 'general')]).transpose(1,2,0), **cfargs)
    
    ##### generate test data ##############################################
    X, I, t_proj, ans, acc = draw_hidden_for_decoding(rnn, [25], nseq, 
                                                      chunking=None,
                                                      decode_from=decode_from, 
                                                      decode_token=decode_token,
                                                      be_picky=None, 
                                                      forget=False, 
                                                      dead_time=None)
    
    ### test peformance ###################################################
    perf[epoch] = np.mean(clf.test(X, ans, t_proj)[:q])
    marg[epoch] = np.mean(clf.margin(X, ans, t_proj)[:q])
    signal_loss[epoch] = np.mean(acc[0][:q])
    noise_loss[epoch] = np.mean(acc[0][q:])
    
    shat_dim[epoch,:] = dclf.test(X, np.array([d for d in  Dichotomies(ans[:,:,:2], 'general')]).transpose(1,2,0)).flatten()
    
    print('done with epoch %d'%epoch)
    
    proj_ctr = clf.project(X, t_proj)
    inner = np.einsum('ik...,jk...->ij...', coefs, coefs)
    
    

#%%
lseq = I.shape[0]
X_ = X.transpose(1,0,2).reshape((X.shape[1],-1))

tim = np.tile(np.arange(lseq)[:,np.newaxis],(1,X.shape[2]))
tim = tim.flatten()
trial = np.tile(np.arange(nseq)[:,np.newaxis],(1,X.shape[0]))
trial = trial.T.flatten() 
token = I.flatten()
#firsthalf = t_proj.flatten() >= L[0]
# whichmemory = ans[:,:,0].flatten() + 2*ans[:,:,1].flatten()
whichmemory = np.sum(ans[:,:,:q]*(2**np.arange(q)),axis=2).flatten()

# do UMAP
#reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.8)
#emb = reducer.fit_transform(X.T)

# do PCA
_, S, V = la.svd(X_[:,:].T-X_[:,:].mean(1), full_matrices=False)
pcs = X_.T@V[:3,:].T

#%% PCA
#plotthese = whichmemory > 0
plotthese = token!=-1
cmap_name = 'nipy_spectral'
# cmap_name = 'viridis'
# colorby = token[plotthese]
#colorby = tim
#colorby = firsthalf[plotthese]
colorby = whichmemory[plotthese]

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

#%%
plotthese = token!=-1
cmap_name = 'nipy_spectral'
# cmap_name = 'viridis'
#colorby = token[plotthese]
#colorby = tim
#colorby = firsthalf[plotthese]
colorby = whichmemory

p1 = proj_ctr[0,:,:].flatten()
p2 = proj_ctr[1,:,:].flatten()
# plot
fig = plt.figure()
scat = plt.scatter(p1, p2, c=colorby, alpha=0.8, cmap=cmap_name)

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.draw_all()
#cb.set_label('Time in trial')

plt.xlabel('Token 0 classifier')
plt.ylabel('Token 1 classifier')
