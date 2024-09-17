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
    input_size = 2
    hidden_sizes = [10,20,30]
    output_size = 1

    model = SimpleFFNN(input_size, hidden_sizes, output_size)

    print(model)

    X = torch.randn(5, input_size)
    output = model(X)

    print(output)
