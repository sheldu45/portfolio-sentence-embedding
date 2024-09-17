import torch
import torch.nn as nn
import torch.optim

class SimpleFFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleFFNN, self).__init__()

        ## Creating layers
        layers = []
        prev_size = input_size

        ## Iterating to create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        ## Adding final layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())

        ## Assembling layers in a nn.Sequential torch object
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple Feedforward Neural Network')
    parser.add_argument('--input_size', type=int, default=2, help='Size of the input layer')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[10, 20, 30], help='Sizes of the hidden layers')
    parser.add_argument('--output_size', type=int, default=1, help='Size of the output layer')

    args = parser.parse_args()

    model = SimpleFFNN(args.input_size, args.hidden_sizes, args.output_size)

    print(model)

    X = torch.randn(5, args.input_size)
    output = model(X)

    print(output)
