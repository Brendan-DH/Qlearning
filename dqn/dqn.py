
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class DeepQNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, n_hidden_layers, nodes_per_layer):
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
            return x

        except RuntimeError as e:
            raise Exception(f"Error occured with DeepQNetwork.forward with the following tensor:\n{x}\n{e}")
