import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RNN(nn.Module):
    """Simple Recurrent Neural Network module.

    This module represents a simple RNN, which consists of multiple layers of RNN cells.

    Args:
        input_dim (int): The dimension of the input vector.
        hidden_dim (int): The dimension of the hidden state vector.
        num_layers (int): The number of layers in the RNN.
        bias (bool): Whether or not to use bias weights. Defaults to True.
        output_dim (int): The dimension of the output vector.
        activation (str): The type of nonlinearity to use. Must be either 'tanh' or 'relu'.
                         Defaults to 'tanh'.
    """

    def __init__(
        self, input_dim, hidden_dim, num_layers, bias, output_dim, activation="tanh"
    ):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bias = bias
        self.output_dim = output_dim

        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.rnn_layers.append(
                    RecurrentNeuralNetworkCell(input_dim, hidden_dim, bias, activation)
                )
            else:
                self.rnn_layers.append(
                    RecurrentNeuralNetworkCell(hidden_dim, hidden_dim, bias, activation)
                )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, initial_hidden_state=None):
        """Forward pass of the RNN.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            initial_hidden_state (torch.Tensor): Initial hidden state tensor of shape (num_layers, batch_size, hidden_dim).

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, output_dim).
        """
        if initial_hidden_state is None:
            initial_hidden_state = torch.zeros(
                self.num_layers, input.size(0), self.hidden_dim
            ).to(device)

        seq_len = input.size(1)
        hidden_states = list(initial_hidden_state)

        for t in range(seq_len):
            for i, layer in enumerate(self.rnn_layers):
                if i == 0:
                    hidden_states[i] = layer(input[:, t, :], hidden_states[i])
                else:
                    hidden_states[i] = layer(hidden_states[i - 1], hidden_states[i])

        last_hidden_state = hidden_states[-1]
        output = self.output_layer(last_hidden_state.squeeze())

        return output
