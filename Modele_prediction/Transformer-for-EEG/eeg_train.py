# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# importing dependencies
from lib.eeg_transformer import *
from lib.train import *

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import detrend, filtfilt, butter, iirnotch, welch
import json
from tqdm import trange


# %%
# torch.nn is a module that implements varios useful functions and functors to implement flexible and highly
# customized neural networks. We will use nn to define neural network modules, different kinds of layers and
# diffrent loss functions
import torch.nn as nn
# torch.nn.functional implements a large variety of activation functions and functional forms of different
# neural network layers. Here we will use it for activation functions.
import torch.nn.functional as F
# torch is the Linear Algebra / Neural Networks library
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# %%
# emg signals profile
seq_len = 20 
emg_channels = 7 # number of electrodes
eeg_size = seq_len*emg_channels*50
embedding_channels = 6125#256 # number of channel in output
embedding_size = seq_len*embedding_channels*3


# %%
class EMGDataset(Dataset):
    def __init__(self, npy_file, experiment='rpoint', params=None, split='train'):        
        # Determine the type of task
        # if experiment in ['rwalk', 'rpoint']:
        #     self.supervised = True
        # elif experiment in ['imagine', 'music', 'speech', 'video']:
        #     self.supervised = False

        data = np.load(npy_file) # Load data
        self.eeg = data
        self.embedding = data

        # if self.supervised == True:
        #     self.embedding = data['E']
        # self.channels = data['channels']  
        self.split = split

        self.supervised = True
        
        if params == None:
            params = {}
            params['num_atoms'] = 1
            params['standardise'] = True
            params['pipeline'] = None
            params['detrend_window'] = 10
            params['sampling_freq'] = 500
            params['line_freq'] = 60
            params['Q_notch'] = 30
            params['low_cutoff_freq'] = 2.
            params['high_cutoff_freq'] = 45.
            params['flatten'] = False
            params['split_ratio'] = [0.7,0,0.3] # train, validation, test
        self.params = params   

        # Split dataset and assign to variables
        self.split_dataset()
        # Change to specified length and reshape
        # if self.eeg.shape[0]%self.params['num_atoms'] != 0:
        #     end = (self.eeg.shape[0]//self.params['num_atoms'])*self.params['num_atoms']
        #     self.eeg = self.eeg[:end]
        #     if self.supervised == True:                
        #         self.embedding = self.embedding[:end]
            
        # new_atom_size = (self.eeg.shape[0]//self.params['num_atoms'])
        # a, b, c = self.eeg.shape
        # self.eeg = self.eeg.reshape(new_atom_size, b*self.params['num_atoms'], c)
        # if self.supervised == True:          
        #     a, b, c = self.embedding.shape
        #     self.embedding = self.embedding.reshape(new_atom_size, b*self.params['num_atoms'], c)         

        self.size = len(self.eeg)
        self.num_channels = self.eeg.shape[2]
        # self.process() # Process parameters and assign to variables

    def split_dataset(self):
        # Determine split indices
        lims = np.dot(self.params['split_ratio'], self.eeg.shape[0])
        lim_ints = [int(lim) for lim in lims]
        lim_ints = np.cumsum(lim_ints)
                
        eeg_sets = {'train': self.eeg[0:lim_ints[0]],
                    'val': self.eeg[lim_ints[0]:lim_ints[1]],
                    'test': self.eeg[lim_ints[1]:]
                   }

        if self.supervised == True:
            emb_sets = {'train': self.embedding[0:lim_ints[0]],
                        'val': self.embedding[lim_ints[0]:lim_ints[1]],
                        'test': self.embedding[lim_ints[1]:]
                       } 

        # Assign particular split
        self.eeg = eeg_sets[self.split]
        # if self.supervised == True:          
        #     self.embedding = emb_sets[self.split]
        return

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg_i = self.eeg[i]
        return torch.from_numpy(eeg_i).float()
        print('a')

    # def process(self):
    #     if self.params['standardise'] == True:
    #         a,b,c = self.eeg.shape
    #         eeg_n = self.eeg.reshape(a*b,c)
    #         mean, std = np.mean(eeg_n, axis = 0), np.std(eeg_n, axis = 0)
    #         eeg_n = (eeg_n-mean)/std
    #         eeg_n = eeg_n.reshape(a,b,c)
    #         self.eeg = eeg_n          
    #     return                  


# %%



# %%
params = {}
params['num_atoms'] = seq_len
params['standardise'] = True
params['pipeline'] = ['rereference', 'detrend', 'bandpassfilter']
params['detrend_window'] = 50
params['sampling_freq'] = 500
params['Q_notch'] = 30
params['low_cutoff_freq'] = 0.1
params['high_cutoff_freq'] = 249.
params['flatten'] = False
params['split_ratio'] = [0.9,0,0.1]

# split training and testing set
batch_size = 10
emg_dataset_train = EMGDataset('stack.npy','rwalk', params, split='train')
emg_dataset_test = EMGDataset('stack.npy','rwalk', params, split='test')

dataloader_train = DataLoader(emg_dataset_train, batch_size=batch_size, shuffle=True)
# eeg_dataset_test = EEGDataset('rwalk.npz','rwalk', params, split='test')
# dataloader_test = DataLoader(eeg_dataset_test, batch_size=batch_size, shuffle=True)


# %%
emg_dataset_train.size


# %%
opt = {}
opt['Transformer-layers'] = 2
opt['Model-dimensions'] = 6125
opt['feedford-size'] = 512
opt['headers'] = 7
opt['dropout'] = 0.1
opt['src_d'] = emg_channels # input dimension
opt['tgt_d'] = embedding_channels # output dimension
opt['timesteps'] = 60


# %%
criterion = nn.MSELoss() # mean squared error
# setup model using hyperparameters defined above
model = make_model(opt['src_d'],opt['tgt_d'],opt['Transformer-layers'],opt['Model-dimensions'],opt['feedford-size'],opt['headers'],opt['dropout'])
# setup optimization function
model_opt = NoamOpt(model_size=opt['Model-dimensions'], factor=1, warmup=400,
        optimizer = torch.optim.Adam(model.parameters(), lr=0.015, betas=(0.9, 0.98), eps=1e-9))
total_epoch = 2000
train_losses = np.zeros(total_epoch)
test_losses = np.zeros(total_epoch)

for epoch in range(total_epoch):
    model.train()
    train_loss = run_epoch(data_gen(dataloader_train), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    train_losses[epoch]=train_loss

    if (epoch+1)%10 == 0:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model_opt.optimizer.state_dict(),
                    'loss': train_loss,
                    }, 'model_checkpoint/'+str(epoch)+'.pth')            
        torch.save(model, 'model_save/model%d.pth'%(epoch)) # save the model

    model.eval() # test the model
    test_loss = run_epoch(data_gen(dataloader_test), model, 
            SimpleLossCompute(model.generator, criterion, None))
    test_losses[epoch] = test_loss
    print('Epoch[{}/{}], train_loss: {:.6f},test_loss: {:.6f}'
              .format(epoch+1, total_epoch, train_loss, test_loss))


# %%
# choose a pair of data from the test set
test_x, test_y = eeg_dataset_test.eeg[1],eeg_dataset_test.embedding[1]
# make a prediction then compare it with its true output
test_out, true_out = output_prediction(model, test_x, test_y, max_len=opt['timesteps'], start_symbol=1,output_d=opt['tgt_d'])


