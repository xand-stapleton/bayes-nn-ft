'''
Project: Bayesian RG and NN-FT
File: bootstrap.py

Code author: Alexander G. Stapleton -- a.g.stapleton@qmul.ac.uk
GitHub handle: xand_stapleton

File map:
---------
This file contains the running and logging code for the simple neural network.
In short, code contained herein is a variant of that employed in the Jupyter
notebook but which can be run non-interactively e.g. for use on the cluster.

Conventions:
------------
Note on convention for Jessica:
I'm using British English so things like optimiser are spelt with an s.

'''

from simple_model import Model
from datasets import MultivariateGaussianDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle

from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PMatDense
from hyperparameters import Hyperparameters
from os import getpid
from time import time

from helper_functions import create_directory

# Parameters
hp = Hyperparameters()

# Data pararameters
means_x = hp.means_x
cov_x = hp.cov_x 
num_train_samples = hp.num_train_samples
num_test_samples = hp.num_test_samples

# RG parameters
num_fisher_samples = hp.num_fisher_samples
fisher_seed = hp.fisher_seed
cutoffs = hp.cutoffs


# Network hyperparameters
train_all_layers = hp.train_all_layers
input_size = hp.input_size 
hidden_size_1 = hp.hidden_size_1 
hidden_size_2 = hp.hidden_size_2 
hidden_size_3 = hp.hidden_size_3 
output_size = hp.output_size 
activation = hp.activation
learnable_func = hp.learnable_func
learning_rate = hp.learning_rate
epochs = hp.epochs
verbosity =  hp.verbosity
train_seed = hp.train_seed
test_seed = hp.test_seed

x_dataset = MultivariateGaussianDataset(num_samples=num_train_samples,
                                        means=means_x, cov=cov_x,
                                        seed=train_seed)

test_x_dataset = MultivariateGaussianDataset(num_samples=num_test_samples,
                                             means=means_x, cov=cov_x,
                                             seed=test_seed)

fisher_x_dataset = MultivariateGaussianDataset(num_fisher_samples,
                                               means=means_x, cov=cov_x,
                                               seed=fisher_seed,
                                               fisher_set=True)

fisher_x_loader = DataLoader(fisher_x_dataset, batch_size=1)

model = Model(input_size, hidden_size_1, hidden_size_2, hidden_size_3, 
              output_size, learnable_func, activation, train_all_layers)

per_epoch_train_loss = model.train(x_dataset, epochs, learning_rate, verbosity)

# Set up history tracking lists
# Evaluations will contain the same test data evaluated at each model cutoff
evaluation_hist = []
killed_param_hist = [[],]
test_loss_hist = []

# Pre-RG evaulations
test_loss, y_pred_hist = model.test(test_x_dataset, track_evals=True)

test_loss_hist.append(test_loss)
evaluation_hist.append(y_pred_hist)

# Cluster run info and save directory
process_pid = getpid()
current_time = time()
output_dir = f'output/model_state_hist_pid-{process_pid}_time-{current_time}'
create_directory(output_dir)

print(evaluation_hist[0][0][0])

# Post-RG evaluations
for idx, cutoff in enumerate(cutoffs):
    cutoff = float(cutoff)
    sloppy_params = model.find_sloppy_params(fisher_x_loader, cutoff).detach().numpy()

    # Only reset the non-killed parameters
    # setdiff1d(x, y) finds parameters in x but not in y (0th run is empty set)
    new_sloppy_params = np.setdiff1d(sloppy_params, killed_param_hist[idx])
    killed_param_hist.append(new_sloppy_params)


    # Re-draw the parameters from their initialisation distribution
    model.reset_params_to_init(new_sloppy_params)

    # Re-evaluate the model with the new sloppy params
    test_loss, y_pred_hist = model.test(test_x_dataset, track_evals=True)

    # Log these to file
    test_loss_hist.append(test_loss)
    evaluation_hist.append(y_pred_hist)
    state_dict_filename = f'{output_dir}/idx-{idx}_cutoff-{cutoff}.pth'
    torch.save(model.model.state_dict(), state_dict_filename)

# Write the run info to file
attributes_file = f'{output_dir}/simp_model_attributes_pid-{process_pid}.pickle'
with open(attributes_file, 'wb') as handle:
    pickle.dump({'hp': hp, 'test_loss_hist': test_loss_hist,
                 'evaluation_hist': evaluation_hist,
                 'killed_param_hist': killed_param_hist}, handle)
