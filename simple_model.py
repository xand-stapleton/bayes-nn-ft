'''
Project: Bayesian RG and NN-FT
File: simple_network.py

Code author: Alexander G. Stapleton -- a.g.stapleton@qmul.ac.uk
GitHub handle: xand_stapleton

File map:
---------
This file contains the simple neural network with the initial layers fixed
at Gaussian initialisation and the final layer trainable.

Conventions:
------------
Note on convention for Jessica:
I'm using British English so things like optimiser are spelt with an s.

'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from optimiser import VGD
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PMatDense

from network import NeuralNetwork, FinalLayer


class Model:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3,
                 output_size, learnable_func, activation=3*[torch.nn.ReLU()],
                 train_all_layers=False):

        # Initialise the frozen layers
        frozen_layers = NeuralNetwork(input_size, hidden_size_1, hidden_size_2,
                                      hidden_size_3, activation=activation)

        # If we're not training all layers, we don't need the frozen layers'
        # gradients
        if not train_all_layers:
            for name, param in frozen_layers.named_parameters():
                if 'weight' in name:
                    param.requires_grad_(False)

        # Final layer
        final_layer = FinalLayer(hidden_size_3, output_size)

        self.model = torch.nn.Sequential(frozen_layers, final_layer)
        print(self.model)

        # Define the loss function (L2 loss)
        self.loss_fn = self._loss

        self.activation = activation

        self.learnable_func = learnable_func

        # These are the size of the weights tensor
        self._output_weights_x = output_size
        self._output_weights_y = hidden_size_3
        self.final_layer_num_params = self._output_weights_x * self._output_weights_y

        # Train all layers (as opposed to just the final one)
        self.train_all_layers = train_all_layers

        # Initialise the optimiser (set in train)
        self.optimiser = None

        # Initialise the diagonal FIM variable
        self.diag_fim = None
        self.final_layer_diag_fim = None

        return None

    def find_sloppy_params(self, fisher_x_loader, cutoff):
        # Check if FIM has already been evaluated (saves re-evaulating)
        if self.diag_fim is None:
            self.diag_fim = FIM(model=self.model, loader=fisher_x_loader,
                                variant='regression', representation=PMatDense,
                                device='cpu', n_output=self._output_weights_x)

            self.diag_fim = self.diag_fim.get_diag()
            self.final_layer_diag_fim = self.diag_fim[-self.final_layer_num_params:].detach().numpy()

        parameter_indices = torch.arange(self.final_layer_num_params)
        sloppy_parameter_indices = parameter_indices[self.final_layer_diag_fim < cutoff]

        return sloppy_parameter_indices

    def reset_params_to_init(self, sloppy_indices):
        # Re-draws sloppy parameters from initialisation distribution
        init_mean = torch.tensor(self.model[1].init_mean, dtype=float)
        init_std = torch.tensor(self.model[1].init_std, dtype=float)

        with torch.no_grad():
            for idx in sloppy_indices:
                column = idx % self._output_weights_y
                row = idx // self._output_weights_y
                self.model[1].final_layer.weight[row, column] = torch.normal(init_mean, init_std)

        return self.model[1].final_layer

    def _loss(self, x, y_pred):
        # Can equivalently use Anindita's cosine loss etc. here
        return nn.MSELoss()(y_pred, self.learnable_func(x))

    def train(self, in_data, epochs, lr=1e-5, verbosity=100):
        # Put the model into training mode
        self.model.train()

        # Define the optimiser (stochastic gradient descent) (only update last
        # layer)
        print('WARNING: Only updating the parameters on the following module: \n', self.model[1])
        if self.train_all_layers:
            self.optimiser = torch.optim.SGD(self.model.parameters(),
                                             lr=lr)
        else:
            self.optimiser = torch.optim.SGD(self.model[1].parameters(),
                                             lr=lr)
        # self.optimiser = torch.optim.Adam(self.model[1].parameters(), lr=lr)
        # self.optimiser = VGD(lr=lr)

        # Training loop
        total_loss = []
        for epoch in range(epochs):
            epoch_loss = []
            for x in in_data:
                # Forward pass
                y_pred = self.model(x)
                loss = self.loss_fn(x, y_pred)
                epoch_loss.append(loss.item())

                # Backward pass and optimization
                # self.optimiser.zero_grad(self.model[1].parameters())
                # loss.backward()
                # self.optimiser.step(self.model[1].parameters())

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            epoch_loss = sum(epoch_loss)/len(epoch_loss)
            total_loss.append(epoch_loss)
            if (epoch+1) % verbosity == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        return total_loss

    def test(self, in_data, track_evals=True):
        # track_evals creates a list of model evaulations
        # Put the model into evaluation mode
        self.model.eval()
        y_pred_hist = []
        test_loss = []
        # Test the trained model
        for x in in_data:
            with torch.no_grad():
                y_pred = self.model(x)
                test_loss.append(self.loss_fn(x, y_pred))

                if track_evals:
                    y_pred_hist.append(y_pred)

        print(f'Test loss average: {sum(test_loss)/len(test_loss):.4f}')

        return test_loss, y_pred_hist
