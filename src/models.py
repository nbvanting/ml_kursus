"""Contains classes of the different neural networks that can be used."""
from typing import Type, Union

import torch.nn as nn
from torch import zeros
from torch.autograd import Variable

from src.model_training import get_activation


# Fully Connected Network Class
class FullyConnectedNetwork(nn.Module):
    """This class represents a fully connected network
    also called a multilayer perceptron.
    Args:
        hidden_size: int: The size of the hidden layers in the network
        num_fcn_layers: int: The number of layers in the network
        input_size: int: The size of the input or number of features in the data
        output_size: int: The size of the output, i.e., the number of steps ahead to predict
        dropout: float: The probability of dropout in the hidden layers
    """

    def __init__(self, input_size: int, **kwargs) -> None:
        super().__init__()

        self.valid_args = [
            "output_size",
            "hidden_size",
            "num_fcn_layers",
            "activation_fnc",
            "dropout",
        ]
        self.args = {}
        for key, value in kwargs.items():
            if key in self.valid_args:
                self.args[key] = value
            else:
                pass

        self.input_size = input_size
        self.activation = get_activation(self.args["activation_fnc"])
        self.flatten = nn.Flatten()
        self.input_layer = nn.Sequential(
            *[nn.Linear(input_size, self.args["hidden_size"]), self.activation()]
        )

        self.hidden_layer = nn.Sequential(
            *[
                nn.Sequential(
                    *[
                        nn.Linear(self.args["hidden_size"], self.args["hidden_size"]),
                        self.activation(),
                        nn.Dropout(p=self.args["dropout"]),
                    ]
                )
                for _ in range(self.args["num_fcn_layers"] - 1)
            ]
        )

        self.output_layer = nn.Linear(
            self.args["hidden_size"], self.args["output_size"]
        )

    def forward(self, x):
        """Forward pass"""
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        return self.output_layer(x)


# LSTM Class
class LongShortTermMemory(nn.Module):
    """This class represents a specialized recurrent neural network
    called long short term memory
    Args:
        hidden_size: int: The size of the hidden layers in the network
        num_rnn_layers: int: The number of stacked recurrent layers in the network
        input_size: int: The size of the input or number of features in the data
        output_size: int: The size of the output, i.e., the number of steps ahead to predict
        dropout: float: The probability of dropout in the hidden layers
        device: str: Device name ['cpu', 'cuda', 'gpu']
    """

    def __init__(self, input_size: int, device: str, **kwargs) -> None:
        super().__init__()

        self.valid_args = ["output_size", "hidden_size", "num_rnn_layers", "dropout"]
        self.args = {}
        for key, value in kwargs.items():
            if key in self.valid_args:
                self.args[key] = value
            else:
                pass

        self.input_size = input_size 
        self.device = device
        self.lstm_layers = nn.LSTM(
            input_size=input_size,
            hidden_size=self.args["hidden_size"],
            num_layers=self.args["num_rnn_layers"],
            dropout=self.args["dropout"],
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            self.args["hidden_size"], self.args["output_size"]
        )

    def forward(self, x):
        """Forward pass"""
        hidden_state = Variable(
            zeros(self.args["num_rnn_layers"], x.size(0), self.args["hidden_size"]).to(
                self.device
            )
        )
        cell_state = Variable(
            zeros(self.args["num_rnn_layers"], x.size(0), self.args["hidden_size"]).to(
                self.device
            )
        )
        out, _ = self.lstm_layers(x, (hidden_state, cell_state))
        out = self.output_layer(out[:, -1, :])
        return out


# GRU Class
class GatedRecurrentUnit(nn.Module):
    """This class represents a specialized recurrent neural network
    called the gated recurrent unit
    Args:
        hidden_size: int: The size of the hidden layers in the network
        num_rnn_layers: int: The number of stacked recurrent layers in the network
        input_size: int: The size of the input or number of features in the data
        output_size: int: The size of the output, i.e., the number of steps ahead to predict
        dropout: float: The probability of dropout in the hidden layers
        device: str: Device name ['cpu', 'cuda', 'gpu']
    """

    def __init__(self, input_size: int, device: str, **kwargs):
        super().__init__()

        self.valid_args = [
            "output_size",
            "hidden_size",
            "num_rnn_layers",
            "dropout",
            "device",
        ]
        self.args = {}
        for key, value in kwargs.items():
            if key in self.valid_args:
                self.args[key] = value
            else:
                pass

        self.input_size = input_size  # required for forward pass
        self.device = device
        # Model
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.args["hidden_size"],
            num_layers=self.args["num_rnn_layers"],
            dropout=self.args["dropout"],
            batch_first=True,
        )
        self.output_layer = nn.Linear(
            self.args["hidden_size"], self.args["output_size"]
        )

    def forward(self, x):
        """Forward pass"""
        h_0 = Variable(
            zeros(self.args["num_rnn_layers"], x.size(0), self.args["hidden_size"]).to(
                self.device
            )
        )
        out, _ = self.gru(x, h_0)
        out = self.output_layer(out[:, -1, :])
        return out


ModelType = Union[
    Type[LongShortTermMemory],
    Type[GatedRecurrentUnit],
    Type[FullyConnectedNetwork],
]


def get_model(model: str) -> ModelType:
    """Function to select a model architecture based on a string
    Accepts the acronyms of the model name -> ['GRU', 'LSTM', 'FCN']
    """
    models = {
        "gru": GatedRecurrentUnit,
        "lstm": LongShortTermMemory,
        "fcn": FullyConnectedNetwork,
    }
    try:
        return models[model.lower()]
    except KeyError as exc:
        raise ValueError(f"Model (acronym) '{model}' not found. Try again.") from exc
