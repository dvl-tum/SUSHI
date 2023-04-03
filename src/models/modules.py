import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, fc_dim, out_dim):
        super(TrajectoryEncoder, self).__init__()
        self.fc_dim = fc_dim
        self.embedding = nn.Linear(input_dim, fc_dim)
        self.encoding = nn.LSTM(input_size=fc_dim, hidden_size=out_dim, num_layers=1, batch_first=True)

    def forward(self, x, x_lens):
        """
        x (num_nodes, track_length, input_dim) will be padded and x_lens will contain the lengths
        """

        n, l, c = x.shape

        x = self.embedding(x.view(-1, c))
        x = x.view(n, l, self.fc_dim)
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        _, (h_t, c_t) = self.encoding(x_packed)

        # return torch.cat((h_t[0], h_t[1]), dim=1)
        return h_t[-1]
