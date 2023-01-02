from .basic_controller import BasicMAC
import torch as th


# This multi-agent controller shares parameters between agents
class TokenMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(TokenMAC, self).__init__(scheme, groups, args)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        if self.agent_output_type == "pi_logits":
            assert False, "unsupported agent_output_type"

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if self.args.agent in ['token_dyan']:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        elif self.args.agent in ['token_updet']:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, 1, -1)
        else:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, self.args.n_tokens, -1)

    def _build_inputs(self, batch, t):
        # currently we only support battles with marines (e.g. 3m 8m 5m_vs_6m)
        # you can implement your own with any other agent type.
        inputs = []
        raw_obs = batch["obs"][:, t]
        reshaped_obs = raw_obs.reshape(-1, self.args.n_tokens, self.args.obs_token_dim)

        inputs.append(reshaped_obs)
        inputs = th.cat(inputs, dim=1)
        mask = None
        return inputs, mask

    def _get_input_shape(self, scheme):
        input_shape = self.args.obs_token_dim
        return input_shape
