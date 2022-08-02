import torch
from torch import nn


class GGNNPropogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """

    def __init__(self, state_dim, n_node):
        super(GGNNPropogator, self).__init__()

        self.n_node = n_node

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        )

    'state_in: the representation of incoming nodes. At the 1st layer, it take embedding matrix'
    'state_out: the representation of outgoing nodes. At the 1st layer, it take embedding matrix'
    'A: the adjacent matrix'

    def forward(self, state_in, state_out, state_cur, A):
        # incoming edges
        a_in = A.T @ state_in
        # outgoing edges
        a_out = A @ state_out
        # concatenate the incoming and outgoing representation
        a = torch.cat((a_in, a_out, state_cur), -1)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), -1)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GlobalAggregator(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """

    def __init__(self, state_dim, n_node, n_steps):
        super(GlobalAggregator, self).__init__()

        self.state_dim = state_dim
        self.n_node = n_node
        self.n_steps = n_steps

        # incoming and outgoing edge embedding
        self.in_fc = nn.Linear(self.state_dim, self.state_dim)
        self.out_fc = nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = GGNNPropogator(self.state_dim, self.n_node)

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, A):
        for i_step in range(self.n_steps):
            in_state = self.in_fc(prop_state)
            out_state = self.out_fc(prop_state)
            prop_state = self.propogator(in_state, out_state, prop_state, A)

        return prop_state
