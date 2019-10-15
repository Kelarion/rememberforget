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
import scipy.special as spc

#%%
class RNNModel(nn.Module):
    """ 
    Skeleton for a couple RNNs 
    Namely: rnn_type = {'LSTM', 'GRU', 'tanh', 'relu', 'tanh-GRU', 'relu-GRU'}
    The final two (tanh-GRU and relu-GRU) use the custom GRU class.
    """
    
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, embed=True):
        super(RNNModel,self).__init__()

        if embed:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = Indicator(ntoken, ninp) # defined below
            
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        elif rnn_type in ['tanh-GRU', 'relu-GRU']:
            nlin = getattr(torch, rnn_type.split('-GRU')[0])
            self.rnn = GatedGRU(ninp, nhid, nlayers, nonlinearity=nlin) # defined below
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
        
    def forward(self, input, hidden, give_gates=False):
        """Only set give_gates=True if it's the custom GRU!!!"""
        if self.embed:
            emb = self.encoder(input)
            if emb.dim()<3:
                emb = emb.unsqueeze(0)
        else:
            emb = input

        if give_gates:
            output, hidden, extras = self.rnn(emb, hidden, give_gates)
        else:
            output, hidden = self.rnn(emb, hidden)

        decoded = self.softmax(self.decoder(output))
        # decoded = self.decoder(output)
        
        if give_gates:
            return decoded, hidden, extras
        else:
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
                                     # hidden = torch.zeros(1,btch.size(1),self.nhid)
                for t in range(btch.size(0)):
                    ignore = (btch[t,...] == padding)
                    #  vals = btch[t:t+1, ...]
                    #  print(torch.sum(ignore))
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Custom GRU by Miguel
class GatedGRU(nn.Module):
    """
    A GRU class which gives access to the gate activations during a forward pass

    Modified from Miguel A Del Rio
    """
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity=torch.tanh):
        """Mimics the nn.GRU module. Currently num_layers is not supported"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(input_size, embedding_size)

        # update gate
        self.W_iu = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hu = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_u = Parameter(torch.Tensor(hidden_size))

        # reset gate
        self.W_ir = Parameter(torch.Tensor(input_size, hidden_size))    
        self.W_hr = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = Parameter(torch.Tensor(hidden_size))

        # hidden state
        self.W_ih = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(torch.Tensor(hidden_size))

        # self.decoder = nn.Linear(hidden_size, output_size)
        self.f = nonlinearity

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    # def init_hidden(self):
    #     h_t = torch.zeros(self.hidden_size)
    #     return h_t

    def forward(self, x, init_state, give_gates=False):
        """Assumes x is of shape (len_seq, batch, input_size)"""
        seq_sz, bs, _ = x.size()

        update_gates = torch.empty(seq_sz, bs, self.hidden_size)
        reset_gates = torch.empty(seq_sz, bs, self.hidden_size)
        hidden_states = torch.empty(seq_sz, bs, self.hidden_size)

        h_t = init_state

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]
            # x_t = self.embedding(x_t)

            z_t = torch.sigmoid(x_t @ self.W_iu + h_t @ self.W_hu + self.b_u)

            r_t = torch.sigmoid(x_t @ self.W_ir + h_t @ self.W_hr + self.b_r)

            h_t = (1 - z_t) * h_t + z_t * self.f(x_t @ self.W_ih + (r_t * h_t) @ self.W_hh + self.b_h)

            update_gates[t,:,:] = z_t
            reset_gates[t,:,:] = r_t
            hidden_states[t,:,:] = h_t

        output = hidden_states

        if give_gates:
            return output, h_t, (update_gates, reset_gates)
        else:
            return output, h_t

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% encoder
class Indicator(nn.Module):
    """
    class to implement indicator function (i.e. one-hot encoding)
    it's a particular type of embedding, which isn't trainable
    """
    def __init__(self, ntoken, ninp):
        super().__init__()
        self.ntoken = ntoken

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
        
        out = np.zeros(x.shape + (self.ntoken,), dtype = int)
        out[all_idx(x, axis=2)] = 1
        out = torch.tensor(out).type(torch.FloatTensor)
        return out

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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

    def is_memory_active(self, seq, mem):
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
                    # plabs = (labs==p).long().squeeze()
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
