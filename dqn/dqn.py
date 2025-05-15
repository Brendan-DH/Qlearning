
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Register buffers for noise
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # Initialize parameters
        self.reset_parameters(sigma_init)

    def reset_parameters(self, sigma_init):
        # Initialize weights and biases
        bound = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

        # Initialize noise scaling factors
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    def forward(self, input):
        # Sample noise
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

        # Compute noisy weights and biases
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        # Perform linear transformation
        return F.linear(input, weight, bias)

class DeepQNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, n_hidden_layers, nodes_per_layer, blocking_function=lambda x: x):
        super(DeepQNetwork, self).__init__()
        # print()
        self.input_layer = nn.Linear(n_observations, nodes_per_layer)
        self.firstHidden = nn.LeakyReLU(1E-4)
        hidden_layers = []
        for i in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            hidden_layers.append(nn.LeakyReLU(1E-4))  # Add ReLU activation after each hidden layer

        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(nodes_per_layer, n_actions)
        self.blocking_function = blocking_function
        self.apply(init_weights)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print("forward x.shape", x, x.shape)
        if x.dtype != torch.float32:
            print("Had to convert the dtype. Original was: ", x.dtype)
            x = x.float()
        if x.device != next(self.parameters()).device:
            print(f"Had to move input device. Original was: {x.device}, now {next(self.parameters()).device})")
            x = x.to(next(self.parameters()).device)
        try:
            x = self.firstHidden(self.input_layer(x))
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
            x = self.output_layer(x)
            x = self.blocking_function(x)
            # print(x)
            return x

        except RuntimeError as e:
            raise Exception(f"Error occured with DeepQNetwork.forward with the following tensor:\n{x}\n{e}")
