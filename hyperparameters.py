import torch
import numpy as np

class Hyperparameters:
    def __init__(self):
        # Data generation parameters
        self.means_x = [1, 1]
        self.cov_x = [[1, 0], [0, 1]]
        # self.num_train_samples = 1500
        self.num_train_samples = 200
        self.num_test_samples = 200
        # Random seed for input data
        self.train_seed = 42 # Stochasticity comes from the init of weights
        # self.train_seed = None # Seed non pull data from /dev/urandom
        self.test_seed = 42 # All test data same
        self.fisher_seed = 42

        # Fisher evaluation parameters
        self.num_fisher_samples = 10
        # This should roughly include each 5th percentile in the lower range
        # (anecdotally true)
        self.cutoffs = torch.tensor([1e-4, 5e-3, 7.5e-3, 1e-2, 5e-2, 7.5e-2,
                                     1e-1, 5e-1, 7.5e-1, 1, 5, 7.5, 10, 50, 75,
                                     100, 5e2, 7.5e2, 1e3, 5e3, 7.5e3, 1e4,
                                     5e4, 7.5e4, 1e5])

        # Network gen. parameters
        self.train_all_layers = False
        self.input_size = len(self.means_x)
        self.hidden_size_1 = 10
        self.hidden_size_2 = 10
        self.hidden_size_3 = 2000
        self.output_size = 1
        self.activation = [torch.nn.Identity(), torch.nn.ReLU(), torch.nn.ReLU()]
        self.learning_rate = 1e-6
        self.epochs = 1000
        self.verbosity = 1 # Number of epochs to log on
        self.learnable_func_noise_std = 1e-6


    def learnable_func(self, x):
        # One common benchmark function is the Rosenbrock problem usually one
        # uses the Rosenrock function for optimiser benchmarking since the
        # minima is hard to find. Instead, we're learning the form. Notice
        # that we add some noise too.
        return torch.tensor([((1 - x[0])**2 - 0.5*(x[1]-(x[0])**2))])
