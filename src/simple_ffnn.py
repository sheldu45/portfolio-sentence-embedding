import torch
import torch.nn as nn
import torch.optim


class SimpleFFNN(nn.Module):
    """A simple feedforward neural network for binary classification.

    This class defines a feedforward neural network (FFNN) with an arbitrary number of hidden layers.
    Each hidden layer uses the ReLU activation function, and the output layer uses the Sigmoid activation function.

    Attributes:
        network (nn.Sequential): A sequential container of the layers in the network.

    Methods:
        __init__(input_size, hidden_sizes, output_size):
            Initializes the SimpleFFNN model with the given layer sizes.
        forward(x):
    """

    def __init__(self, input_size, hidden_sizes, output_size, init_method='kaiming'):
        """
        Initializes the SimpleFFNN model.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (list of int): Sizes of the hidden layers.
            output_size (int): Size of the output layer.
        """
        super(SimpleFFNN, self).__init__()

        # Creating layers
        layers = []
        prev_size = input_size

        # Iterating to create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Adding the final output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())

        # Assembling layers in a nn.Sequential torch object
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.network(x)
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple Feedforward Neural Network')
    parser.add_argument('--input_size', type=int, default=2, help='Size of the input layer')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[10, 20, 30],
                        help='Sizes of the hidden layers')
    parser.add_argument('--output_size', type=int, default=1, help='Size of the output layer')

    args = parser.parse_args()

    # Initialize the model with the provided arguments
    model = SimpleFFNN(args.input_size, args.hidden_sizes, args.output_size)

    print(model)

    # Generate a random input tensor
    X = torch.randn(5, args.input_size)

    # Get the model output
    output = model(X)

    print(output)
