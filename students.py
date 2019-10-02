"""
Classes used in the Remember-Forget experiments.
 
Includes:
    - RNNModel: class that currently supports {'LSTM', 'GRU', 'tanh', 'relu'}
    recurrent neural networks. includes a save, load, and train method.
    - stateDecoder: does the memory decoding. includes a train method.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.special as spc

#%%
class RNNModel(nn.Module):
    """ 
    Skeleton for a couple RNNs 
    Namely: rnn_type = {'LSTM', 'GRU', 'tanh', 'relu'}
    """
    
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, embed=True):
        super(RNNModel,self).__init__()
        
        if embed:
            self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
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
        
    def init_weights(self):
        if self.embed:
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_()
        
    def forward(self, input, hidden):
        if self.embed:
            emb = self.encoder(input)
            if emb.dim()<3:
                emb = emb.unsqueeze(0)
        else:
            emb = input
        output, hidden = self.rnn(emb, hidden)
        decoded = self.softmax(self.decoder(output))
#        decoded = self.decoder(output)
    
        return decoded, hidden
    
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
        ToDo: support give_output=True
        """
        
        self.optimizer = algo(self.parameters(), **optparams)
    
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
#                hidden = torch.zeros(1,btch.size(1),self.nhid)
                for t in range(btch.size(0)):
                    ignore = (btch[t,...] == padding)
#                    vals = btch[t:t+1, ...]
#                    print(torch.sum(ignore))
                    out, hidden = self(btch[t:t+1, ...], hidden)
                    output[~ignore,:] = out[0, ~ignore, :]
                
                loss = criterion(output.squeeze(0),labs.long().squeeze()) # | || || |_
                loss.backward()
                
                self.optimizer.step()
                
                # update loss
                running_loss += loss.item()
                loss_ = np.append(loss_, loss.item())
                
            if (epsilon>0) and (np.abs(running_loss-prev_loss) <= epsilon):
                print('~'*5)
                print('[%d] Converged at loss = %0.3f'%(epoch+1, running_loss/i))
                return loss_
            # print to screen
            if do_print:
                print('[%d] loss: %.3f' %
                      (epoch + 1, running_loss / i))
                running_loss = 0.0
                prev_loss = running_loss
                
        print('[%d] Finished at loss = %0.3f'%(epoch+1, running_loss/i))
        print('~'*5)
        return loss_
        
#class RemForTask:
#    """
#    Class for storing parameters of a remember-forget task
#    The idea is these parameters should be sufficient to identify an experiment
#    """
#    def __init__(self, datamaker, dmargs):
#        """
#        datamaker (callable): function which produces data for the task
#            - must output arrays X, Y: data, labels
#        dmargs (dict): all arguments passed into datamaker
#            example for the standard task:
#            - L: number of tokens (int)
#            - nseq: maximum number of sequences (int)
#            - AB: alphabet (list)
#            - forget_switch: token to signal forgetting (int or str)
#        
#        ToDo: include RNN and optimization parameters 
#        """
#        
#        self.datamaker = datamaker
#        
#        self.L = L
#        self.AB = AB
#        self.switch = forget_switch
#        self.nseq = nseq
#        
#        self.ninp = len(AB)
#        self.lenseq = 2*L-1
#        if forget_switch is not None:
#            self.lenseq += 1
#            self.ninp += 1

#%% decoder
class stateDecoder(nn.Module):
    """
    For getting the probability that each token is currently represented
    Right now all decoders are separate 
    """
    
    def __init__(self, tokens, nhid):
        super(stateDecoder,self).__init__()
        
        ntoken = len(tokens)
        for p in range(ntoken): # this is kind of terrible (??)
            setattr(self, 'decoder%d'%(p), nn.Linear(nhid, 2))
            setattr(self, 'softmax%d'%(p), nn.LogSoftmax(dim=1))
        
        self.ntoken = ntoken
        self.tokens = tokens
        self.nhid = nhid
        
        self.init_weights()
        
    def init_weights(self):
        for p in range(self.ntoken): # yiiiiiikes
            getattr(self, 'decoder%d'%(p)).bias.data.zero_()
            getattr(self, 'decoder%d'%(p)).weight.data.uniform_()
        
    def forward(self, hidden):
        
        decoded = [[] for _ in range(self.ntoken)]
        for p in range(self.ntoken): # oooohhhhh my god this is bad
            decoded[p] = getattr(self, 'softmax%d'%(p))(getattr(self, 'decoder%d'%(p))(hidden))
        
        return decoded
    
    def is_memory_active(seq, mem):
        """
        Returns step function for whether mem is 'active' in seq
        """
        stp = np.cumsum((seq==mem).astype(int), axis = 1) % 2
        return stp
    
    def train(self, X, Y, optparams, dlparams, algo=optim.SGD,
              criterion = nn.NLLLoss(), nepoch=1000, do_print=True,
              epsilon = 0):
        """
        Train rnn on data X and labels Y (both torch tensors).
        X, Y need to have samples as the FIRST dimension
        ToDo: support give_output=True
        """
        
        self.optimizer = algo(self.parameters(), **optparams)
        
        dset = torch.utils.data.TensorDataset(X,Y)
        trainloader = torch.utils.data.DataLoader(dset, **dlparams)
        
        loss_ = np.zeros((0, self.ntoken))
        prev_loss =  np.zeros(self.ntoken)
        for epoch in range(nepoch):
            running_loss = np.zeros(self.ntoken)
            
            for i, batch in enumerate(trainloader, 0):
                # unpack
                btch, labs = batch
                
                # initialise
                self.optimizer.zero_grad()
                
                # forward -> backward -> optimize
                decoded = self(btch) 
                
                # | || || |_
                loss = [[] for _ in range(self.ntoken)]
                for t, p in enumerate(self.tokens):
                    plabs = self.is_memory_active(labs, p)
                    plabs = plabs.view(-1,1).long().squeeze()
#                    plabs = (labs==p).long().squeeze()
                    loss[t] = criterion(decoded[t], plabs)
                    loss[t].backward()
                
                self.optimizer.step()
                
                # update loss
                running_loss += [l.item() for l in loss]
                loss_ = np.append(loss_, [[l.item() for l in loss]], axis=0)
                
            if (epsilon>0) and np.all(np.abs(running_loss-prev_loss) <= epsilon):
                print('[%d] Converged at loss = %0.3f' % 
                      (epoch+1, str((running_loss/i).round(3))))
                print('~'*5)
                return loss_
            
            # print to screen
            if do_print:
                print('[%d] loss: %s' %
                      (epoch + 1, str((running_loss/i).round(3))))
                running_loss = np.zeros(self.ntoken)
            
        print('[%d] Finished at loss = %0.3f'%(epoch+1, running_loss/i))
        print('~'*5)
        
        return loss_


