import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import TokenOPTTransformer


class TokenOPTAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TokenOPTAgent, self).__init__()
        self.args = args
        self.x_shape = None

        self.token_embedding = nn.Linear(input_shape, args.emb_dim)
        self.transformer = TokenOPTTransformer(args.n_blocks, args.emb_dim, args.n_heads, args.emb_dim * 4, args.rnn_hidden_dim)

        self.fc1 = nn.Linear(args.emb_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_fixed_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x, mask = inputs
        self.x_shape = x.size()
        b, t, e = x.size()

        x = F.relu(self.token_embedding(x))
        x = self.transformer.forward(x, hidden_state.reshape(-1, t, self.args.rnn_hidden_dim), mask)

        x = F.relu(self.fc1(x)).reshape(-1, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in).reshape(b, t, self.args.rnn_hidden_dim)

        q_self_actions = self.fc2(h[:, 0, :])

        q = q_self_actions
        q_mutual_actions = self.fc2(h[:, 1:self.args.n_enemies + 1, :]).mean(2)
        q = torch.cat((q, q_mutual_actions), 1)

        return q, h

    def get_disentangle_loss(self):
        b, t, e = self.x_shape

        loss = 0
        for block in self.transformer.transformer_blocks:
            loss += block.attn.cal_disentangle_loss()
        loss = torch.mean(loss.reshape(-1, b, t).permute(1, 0, 2), dim=2)

        return loss

    def get_cmi_loss(self):
        b, t, e = self.x_shape

        entropy_loss = 0
        kl_loss = 0
        for block in self.transformer.transformer_blocks:
            loss1, loss2 = block.attn.cal_cmi_loss()
            entropy_loss += loss1
            kl_loss += loss2
        entropy_loss = entropy_loss.reshape(-1, b).permute(1, 0)
        kl_loss = kl_loss.reshape(-1, b).permute(1, 0)

        return entropy_loss, kl_loss

    def set_pattern(self, use_pattern):
        for block in self.transformer.transformer_blocks:
            block.attn.set_pattern(use_pattern=use_pattern)

