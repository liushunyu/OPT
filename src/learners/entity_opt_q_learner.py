import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.entity_opt_qmix import EntityOPTQMixer
import torch as th
from torch.optim import RMSprop, Adam


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "entity_opt_qmix":
                self.mixer = EntityOPTQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def _build_inputs(self, batch):
        entities = []
        bs, ts, ne, ed = batch["entities"].shape
        entities.append(batch["entities"])
        if self.args.entity_last_action:
            last_actions = th.zeros(bs, ts, ne, self.args.n_actions,
                                    device=batch.device, dtype=batch["entities"].dtype)
            last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
            entities.append(last_actions)
        entities = th.cat(entities, dim=3)

        inputs = (entities[:, :-1], batch["entity_mask"][:, :-1])
        target_inputs = (entities[:, 1:], batch["entity_mask"][:, 1:])
        return inputs, target_inputs

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # mask the last_data after terminated
        avail_actions = batch["avail_actions"]

        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.mixer.train()
        self.target_mac.eval()
        self.target_mixer.eval()

        self.mac.agent.set_pattern(use_pattern=True)

        # Calculate estimated Q-Values
        self.mac.init_hidden(batch.batch_size)
        mac_out = self.mac.forward(batch, t=None)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.init_hidden(batch.batch_size)
            target_mac_out = self.target_mac.forward(batch, t=None)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = target_mac_out[:, 1:]

            # Mask out unavailable actions
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            # Max over target Q-Values
            if self.args.double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            inputs, target_inputs = self._build_inputs(batch)
            chosen_action_qvals = self.mixer(chosen_action_qvals, inputs)
            target_max_qvals = self.target_mixer(target_max_qvals, target_inputs)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = td_loss

        mixer_disentangle_loss = self.mixer.get_disentangle_loss()
        mixer_disentangle_loss = (mixer_disentangle_loss * mask.squeeze(2)).sum() / mask.sum()
        loss = loss + self.args.mixer_disentangle_alpha * mixer_disentangle_loss

        mac_disentangle_loss = self.mac.agent.get_disentangle_loss()
        mac_disentangle_loss = (mac_disentangle_loss[:, :-1] * mask.squeeze(2)).sum() / mask.sum()
        loss = loss + self.args.mac_disentangle_alpha * mac_disentangle_loss

        mac_cmi_entropy_loss, mac_cmi_kl_loss = self.mac.agent.get_cmi_loss()
        mac_cmi_entropy_loss = (mac_cmi_entropy_loss[:, :-1] * mask.squeeze(2)).sum() / mask.sum()
        mac_cmi_kl_loss = (mac_cmi_kl_loss[:, :-1] * mask.squeeze(2)).sum() / mask.sum()
        loss = loss + self.args.mac_cmi_entropy_alpha * mac_cmi_entropy_loss
        loss = loss + self.args.mac_cmi_kl_alpha * mac_cmi_kl_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.mac.agent.set_pattern(use_pattern=False)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("train/loss", loss.item(), t_env)
            self.logger.log_stat("train/td_loss", td_loss.item(), t_env)
            self.logger.log_stat("train/grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("train/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("train/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("train/target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
