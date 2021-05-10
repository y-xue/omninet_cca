import torch

class FFN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(FFN, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear1(x)
        