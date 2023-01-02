import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityOPTTransformer


class EntityOPTAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EntityOPTAgent, self).__init__()
        self.args = args
        self.entity_shape = None

        self.entity_embedding = nn.Linear(input_shape, args.emb_dim)
        self.transformer = EntityOPTTransformer(args.n_blocks, args.emb_dim, args.n_heads, args.emb_dim * 4, args.rnn_hidden_dim)
        self.fc1 = nn.Linear(args.emb_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        entities, obs_mask, entity_mask = inputs
        self.entity_shape = entities.shape
        b, s, t, e = entities.shape
        entities = entities.reshape(b * s, t, e)
        obs_mask = obs_mask.reshape(b * s, t, t)
        entity_mask = entity_mask.reshape(b * s, t)
        agent_mask = entity_mask[:, :self.args.n_agents]

        x = F.relu(self.entity_embedding(entities))
        x = x.reshape(b, s, t, -1)
        obs_mask = obs_mask.reshape(b, s, t, t)
        agent_mask = agent_mask.reshape(b, s, self.args.n_agents)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = []
        for i in range(s):
            out = self.transformer(x[:, i], h_in.reshape(b, self.args.n_agents, self.args.rnn_hidden_dim), obs_mask[:, i])[:, :self.args.n_agents]
            out = out.masked_fill(agent_mask[:, i].unsqueeze(2), 0)

            out = F.relu(self.fc1(out))

            h_in = self.rnn(out.reshape(-1, self.args.rnn_hidden_dim), h_in)
            h.append(h_in.reshape(b, self.args.n_agents, self.args.rnn_hidden_dim))
        h = torch.stack(h, dim=1)

        q = self.fc2(h)
        q = q.reshape(b, s, self.args.n_agents, -1)
        q = q.masked_fill(agent_mask.reshape(b, s, self.args.n_agents, 1), 0)

        return q, h

    def get_disentangle_loss(self):
        b, s, t, e = self.entity_shape

        loss = 0
        for block in self.transformer.transformer_blocks:
            loss += block.attn.cal_disentangle_loss()
        loss = torch.mean(loss.reshape(s, b, t).permute(1, 0, 2), dim=2)

        return loss

    def get_cmi_loss(self):
        b, s, t, e = self.entity_shape

        entropy_loss = 0
        kl_loss = 0
        for block in self.transformer.transformer_blocks:
            loss1, loss2 = block.attn.cal_cmi_loss()
            entropy_loss += loss1
            kl_loss += loss2
        entropy_loss = torch.mean(entropy_loss.reshape(s, b, -1).permute(1, 0, 2), dim=2)
        kl_loss = torch.mean(kl_loss.reshape(s, b, -1).permute(1, 0, 2), dim=2)

        return entropy_loss, kl_loss

    def set_pattern(self, use_pattern):
        for block in self.transformer.transformer_blocks:
            block.attn.set_pattern(use_pattern=use_pattern)
