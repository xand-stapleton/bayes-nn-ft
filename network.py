import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3,
                 activation):
        super(NeuralNetwork, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, hidden_size_1, bias=False)
        self.hidden_layer_2 = nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        self.hidden_layer_3 = nn.Linear(hidden_size_2, hidden_size_3, bias=False)

        self.activation_1 = activation[0]
        self.activation_2 = activation[1]
        self.activation_3 = activation[2]

        # Initialize weights from a normal distribution
        nn.init.normal_(self.hidden_layer_1.weight, mean=0, std=1)
        nn.init.normal_(self.hidden_layer_2.weight, mean=0, std=1)
        nn.init.normal_(self.hidden_layer_3.weight, mean=0, std=1)


    def forward(self, x):
        x = self.activation_1(self.hidden_layer_1(x))
        x = self.activation_2(self.hidden_layer_2(x))
        x = self.activation_3(self.hidden_layer_3(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, input_size, output_size, init_mean=0, init_std=1):
        super(FinalLayer, self).__init__()
        self.init_mean = init_mean
        self.init_std = init_std
        self.final_layer = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.final_layer.weight, mean=init_mean, std=init_std)

    def forward(self, x):
        x = self.final_layer(x)
        return x
