import numpy as np
from numpy.random import RandomState
from ..multiagentenv import MultiAgentEnv


class PreyEnv(MultiAgentEnv):
    def __init__(self, map_name="various_all", entity_scheme=True, map_size=12, sight_range=4,
                 episode_limit=50, seed=None):
        super(PreyEnv, self).__init__()

        self.map_name = map_name
        self.entity_scheme = entity_scheme
        self.n_actions = 6  # up, down, left, right, stay, capture

        if self.map_name == 'various_num':
            self.n_agents = 9
            self.min_n_agents = 2
            self.max_n_agents = 9
            self.min_n_preys = 2
            self.max_n_preys = 9
            self.min_n_landmarks = 2
            self.max_n_landmarks = 5
        elif self.map_name == 'various_cap':
            self.n_agents = 5
            self.min_n_agents = 5
            self.max_n_agents = 5
            self.min_n_preys = 5
            self.max_n_preys = 5
            self.min_n_landmarks = 4
            self.max_n_landmarks = 4
        elif self.map_name == 'various_all':
            self.n_agents = 9
            self.min_n_agents = 2
            self.max_n_agents = 9
            self.min_n_preys = 2
            self.max_n_preys = 9
            self.min_n_landmarks = 2
            self.max_n_landmarks = 5

        self.map_size = map_size
        self.sight_range = sight_range

        self.episode_limit = episode_limit

        self._seed = seed
        self.rs = RandomState(seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(u) for u in actions[:self.n_agents]]

        prey_heals = self.prey_caps.copy()

        for a, u in enumerate(actions):
            # stay, up, down, left, right, capture
            if u in [1, 2, 3, 4]:
                self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1])] = 0
                self.agent_locs[a, int(self.agent_coords[a, 0] * self.map_size + self.agent_coords[a, 1])] = 0

                if u == 1:
                    # up
                    if self.map[int(self.agent_coords[a, 0]) - 1, int(self.agent_coords[a, 1])] == 0:
                        self.agent_coords[a, 0] = self.agent_coords[a, 0] - 1
                elif u == 2:
                    # down
                    if self.map[int(self.agent_coords[a, 0]) + 1, int(self.agent_coords[a, 1])] == 0:
                        self.agent_coords[a, 0] = self.agent_coords[a, 0] + 1
                elif u == 3:
                    # left
                    if self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1]) - 1] == 0:
                        self.agent_coords[a, 1] = self.agent_coords[a, 1] - 1
                elif u == 4:
                    # right
                    if self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1]) + 1] == 0:
                        self.agent_coords[a, 1] = self.agent_coords[a, 1] + 1

                self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1])] = a + 1
                self.agent_locs[a, int(self.agent_coords[a, 0] * self.map_size + self.agent_coords[a, 1])] = 1
            elif u == 5:
                if self.agent_coords[a, 0] != 0:
                    if -100 < self.map[int(self.agent_coords[a, 0]) - 1, int(self.agent_coords[a, 1])] < 0:
                        prey_heals[-int(self.map[int(self.agent_coords[a, 0]) - 1, int(self.agent_coords[a, 1])]) - 1] -= self.agent_caps[a]

                if self.agent_coords[a, 0] != self.map_size - 1:
                    if -100 < self.map[int(self.agent_coords[a, 0]) + 1, int(self.agent_coords[a, 1])] < 0:
                        prey_heals[-int(self.map[int(self.agent_coords[a, 0]) + 1, int(self.agent_coords[a, 1])]) - 1] -= self.agent_caps[a]

                if self.agent_coords[a, 1] != 0:
                    if -100 < self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1]) - 1] < 0:
                        prey_heals[-int(self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1]) - 1]) - 1] -= self.agent_caps[a]

                if self.agent_coords[a, 1] != self.map_size - 1:
                    if -100 < self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1]) + 1] < 0:
                        prey_heals[-int(self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1]) + 1]) - 1] -= self.agent_caps[a]

        reward = 0.0
        for p in range(self.n_preys):
            if self.prey_caps[p] == 0:
                continue

            if prey_heals[p] <= 0:
                self.prey_caps[p] = 0
                self.map[int(self.prey_coords[p, 0]), int(self.prey_coords[p, 1])] = 0
                self.prey_locs[p, int(self.prey_coords[p, 0] * self.map_size + self.prey_coords[p, 1])] = 0
                self.prey_coords[p, 0] = 0
                self.prey_coords[p, 1] = 0
                reward += 1.0
            else:
                self.map[int(self.prey_coords[p, 0]), int(self.prey_coords[p, 1])] = 0
                self.prey_locs[p, int(self.prey_coords[p, 0] * self.map_size + self.prey_coords[p, 1])] = 0

                u = self.rs.randint(0, 5)  # stay, up, down, left, right
                if u == 1:
                    # up
                    if self.prey_coords[p, 0] != 0:
                        if self.map[int(self.prey_coords[p, 0]) - 1, int(self.prey_coords[p, 1])] == 0:
                            self.prey_coords[p, 0] = self.prey_coords[p, 0] - 1
                elif u == 2:
                    # down
                    if self.prey_coords[p, 0] != self.map_size - 1:
                        if self.map[int(self.prey_coords[p, 0]) + 1, int(self.prey_coords[p, 1])] == 0:
                            self.prey_coords[p, 0] = self.prey_coords[p, 0] + 1
                elif u == 3:
                    # left
                    if self.prey_coords[p, 1] != 0:
                        if self.map[int(self.prey_coords[p, 0]), int(self.prey_coords[p, 1]) - 1] == 0:
                            self.prey_coords[p, 1] = self.prey_coords[p, 1] - 1
                elif u == 4:
                    # right
                    if self.prey_coords[p, 1] != self.map_size - 1:
                        if self.map[int(self.prey_coords[p, 0]), int(self.prey_coords[p, 1]) + 1] == 0:
                            self.prey_coords[p, 1] = self.prey_coords[p, 1] + 1

                self.map[int(self.prey_coords[p, 0]), int(self.prey_coords[p, 1])] = -p - 1
                self.prey_locs[p, int(self.prey_coords[p, 0] * self.map_size + self.prey_coords[p, 1])] = 1

        info = {'battle_won': False}
        done = False
        if self.prey_caps.sum() == 0:
            done = True
            info['battle_won'] = True

        self.t += 1
        if self.t == self.episode_limit:
            done = True
            info['episode_limit'] = True

        return reward, done, info

    def _calc_distance_mtx(self):
        dist_mtx = 1000 * np.ones((self.n_agents + self.n_preys + self.n_landmarks, self.n_agents + self.n_preys + self.n_landmarks))
        for i in range(self.n_agents + self.n_preys + self.n_landmarks):
            for j in range(self.n_agents + self.n_preys + self.n_landmarks):
                if j < i:
                    continue
                elif j == i:
                    dist_mtx[i, j] = 0.0
                else:
                    is_dead_prey_0 = False
                    is_dead_prey_1 = False
                    if i >= self.n_agents + self.n_preys:
                        x_0 = self.landmark_coords[i - self.n_agents - self.n_preys, 0]
                        y_0 = self.landmark_coords[i - self.n_agents - self.n_preys, 1]
                    elif i >= self.n_agents:
                        if self.prey_caps[i - self.n_agents] == 0:
                            is_dead_prey_0 = True
                        x_0 = self.prey_coords[i - self.n_agents, 0]
                        y_0 = self.prey_coords[i - self.n_agents, 1]
                    else:
                        x_0 = self.agent_coords[i, 0]
                        y_0 = self.agent_coords[i, 1]

                    if j >= self.n_agents + self.n_preys:
                        x_1 = self.landmark_coords[j - self.n_agents - self.n_preys, 0]
                        y_1 = self.landmark_coords[j - self.n_agents - self.n_preys, 1]
                    elif j >= self.n_agents:
                        if self.prey_caps[j - self.n_agents] == 0:
                            is_dead_prey_1 = True
                        x_1 = self.prey_coords[j - self.n_agents, 0]
                        y_1 = self.prey_coords[j - self.n_agents, 1]
                    else:
                        x_1 = self.agent_coords[j, 0]
                        y_1 = self.agent_coords[j, 1]

                    if not is_dead_prey_0 and not is_dead_prey_1:
                        dist = np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)
                        dist_mtx[i, j] = dist
                        dist_mtx[j, i] = dist

        return dist_mtx

    def get_masks(self):
        dist_mtx = self._calc_distance_mtx()
        obs_mask = (dist_mtx > self.sight_range).astype(np.bool8)

        obs_mask_pad = np.ones((self.max_n_agents + self.max_n_preys + self.max_n_landmarks,
                                self.max_n_agents + self.max_n_preys + self.max_n_landmarks), dtype=np.uint8)

        obs_mask_pad[:self.n_agents, :self.n_agents] = (
            obs_mask[:self.n_agents, :self.n_agents]
        )
        obs_mask_pad[:self.n_agents, self.max_n_agents:self.max_n_agents + self.n_preys] = (
            obs_mask[:self.n_agents, self.n_agents:self.n_agents + self.n_preys]
        )
        obs_mask_pad[:self.n_agents, self.max_n_agents + self.max_n_preys:self.max_n_agents + self.max_n_preys + self.n_landmarks] = (
            obs_mask[:self.n_agents, self.n_agents + self.n_preys:self.n_agents + self.n_preys + self.n_landmarks]
        )

        obs_mask_pad[self.max_n_agents:self.max_n_agents + self.n_preys, :self.n_agents] = (
            obs_mask[self.n_agents:self.n_agents + self.n_preys, :self.n_agents]
        )
        obs_mask_pad[self.max_n_agents:self.max_n_agents + self.n_preys, self.max_n_agents:self.max_n_agents + self.n_preys] = (
            obs_mask[self.n_agents:self.n_agents + self.n_preys, self.n_agents:self.n_agents + self.n_preys]
        )
        obs_mask_pad[self.max_n_agents:self.max_n_agents + self.n_preys, self.max_n_agents + self.max_n_preys:self.max_n_agents + self.max_n_preys + self.n_landmarks] = (
            obs_mask[self.n_agents:self.n_agents + self.n_preys, self.n_agents + self.n_preys:self.n_agents + self.n_preys + self.n_landmarks]
        )

        obs_mask_pad[self.max_n_agents + self.max_n_preys:self.max_n_agents + self.max_n_preys + self.n_landmarks, :self.n_agents] = (
            obs_mask[self.n_agents + self.n_preys:self.n_agents + self.n_preys + self.n_landmarks, :self.n_agents]
        )
        obs_mask_pad[self.max_n_agents + self.max_n_preys:self.max_n_agents + self.max_n_preys + self.n_landmarks, self.max_n_agents:self.max_n_agents + self.n_preys] = (
            obs_mask[self.n_agents + self.n_preys:self.n_agents + self.n_preys + self.n_landmarks, self.n_agents:self.n_agents + self.n_preys]
        )
        obs_mask_pad[self.max_n_agents + self.max_n_preys:self.max_n_agents + self.max_n_preys + self.n_landmarks, self.max_n_agents + self.max_n_preys:self.max_n_agents + self.max_n_preys + self.n_landmarks] = (
            obs_mask[self.n_agents + self.n_preys:self.n_agents + self.n_preys + self.n_landmarks, self.n_agents + self.n_preys:self.n_agents + self.n_preys + self.n_landmarks]
        )

        entity_mask = np.ones(self.max_n_agents + self.max_n_preys + self.max_n_landmarks, dtype=np.uint8)
        entity_mask[:self.n_agents] = 0
        entity_mask[self.max_n_agents:self.max_n_agents + self.n_preys] = 0
        entity_mask[self.max_n_agents + self.max_n_preys:self.max_n_agents + self.max_n_preys + self.n_landmarks] = 0
        return obs_mask_pad, entity_mask

    def get_entities(self):
        agent_entities = np.concatenate((self.agent_type, self.agent_ids, self.agent_locs, self.agent_caps), axis=1).copy()
        prey_entities = np.concatenate((self.prey_type, self.prey_ids, self.prey_locs, self.prey_caps), axis=1).copy()
        landmark_entities = np.concatenate((self.landmark_type, self.landmark_ids, self.landmark_locs, self.landmark_caps), axis=1).copy()

        entities = [agent_entities[i] for i in range(self.n_agents)] + [np.zeros(self.get_entity_size(), dtype=np.float32) for _ in range(self.max_n_agents - self.n_agents)] +\
                   [prey_entities[i] for i in range(self.n_preys)] + [np.zeros(self.get_entity_size(), dtype=np.float32) for _ in range(self.max_n_preys - self.n_preys)] +\
                   [landmark_entities[i] for i in range(self.n_landmarks)] + [np.zeros(self.get_entity_size(), dtype=np.float32) for _ in range(self.max_n_landmarks - self.n_landmarks)]

        return entities

    def get_entity_size(self):
        return 3 + self.max_n_agents + self.map_size * self.map_size + 1

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return np.concatenate(self.get_entities())

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.get_entity_size() * self.n_agents

    def get_state(self):
        return np.concatenate(self.get_entities())

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_entity_size() * self.n_agents

    def get_avail_actions(self):
        # stay, up, down, left, right, capture
        avail_actions = [[1, 1, 1, 1, 1, 0] for _ in range(self.n_agents)] + [[1, 0, 0, 0, 0, 0] for _ in range(self.max_n_agents - self.n_agents)]
        for a in range(self.n_agents):
            if self.agent_coords[a, 0] == 0 or self.map[int(self.agent_coords[a, 0]) - 1, int(self.agent_coords[a, 1])] != 0:
                # up
                avail_actions[a][1] = 0

            if self.agent_coords[a, 0] == self.map_size - 1 or self.map[int(self.agent_coords[a, 0]) + 1, int(self.agent_coords[a, 1])] != 0:
                # down
                avail_actions[a][2] = 0

            if self.agent_coords[a, 1] == 0 or self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1]) - 1] != 0:
                # left
                avail_actions[a][3] = 0

            if self.agent_coords[a, 1] == self.map_size - 1 or self.map[int(self.agent_coords[a, 0]), int(self.agent_coords[a, 1]) + 1] != 0:
                # right
                avail_actions[a][4] = 0

        for p in range(self.n_preys):
            if self.prey_caps[p] == 0:
                continue

            if self.prey_coords[p, 0] != 0:
                if self.map[int(self.prey_coords[p, 0]) - 1, int(self.prey_coords[p, 1])] > 0:
                    avail_actions[int(self.map[int(self.prey_coords[p, 0]) - 1, int(self.prey_coords[p, 1])]) - 1][5] = 1

            if self.prey_coords[p, 0] != self.map_size - 1:
                if self.map[int(self.prey_coords[p, 0]) + 1, int(self.prey_coords[p, 1])] > 0:
                    avail_actions[int(self.map[int(self.prey_coords[p, 0]) + 1, int(self.prey_coords[p, 1])]) - 1][5] = 1

            if self.prey_coords[p, 1] != 0:
                if self.map[int(self.prey_coords[p, 0]), int(self.prey_coords[p, 1]) - 1] > 0:
                    avail_actions[int(self.map[int(self.prey_coords[p, 0]), int(self.prey_coords[p, 1]) - 1]) - 1][5] = 1

            if self.prey_coords[p, 1] != self.map_size - 1:
                if self.map[int(self.prey_coords[p, 0]), int(self.prey_coords[p, 1]) + 1] > 0:
                    avail_actions[int(self.map[int(self.prey_coords[p, 0]), int(self.prey_coords[p, 1]) + 1]) - 1][5] = 1

        return avail_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return {}

    def reset(self, **kwargs):
        self.test_mode = kwargs['test_mode']
        if self.map_name == 'various_num':
            if not self.test_mode:
                self.n_agents = self.rs.choice([2, 3, 5, 8], 1, replace=False)[0]
                self.n_preys = self.rs.choice([2, 3, 5, 8], 1, replace=False)[0]
                self.n_landmarks = self.rs.choice([2, 4], 1, replace=False)[0]
            else:
                self.n_agents = self.rs.choice([4, 6, 7, 9], 1, replace=False)[0]
                self.n_preys = self.rs.choice([4, 6, 7, 9], 1, replace=False)[0]
                self.n_landmarks = self.rs.choice([3, 5], 1, replace=False)[0]

            self.agent_caps = np.zeros((self.n_agents, 1), dtype=np.float32) + 1
            self.prey_caps = np.zeros((self.n_preys, 1), dtype=np.float32) + 2
            self.landmark_caps = np.zeros((self.n_landmarks, 1), dtype=np.float32)

        elif self.map_name == 'various_cap':
            self.n_agents = 5
            self.n_preys = 5
            self.n_landmarks = 4

            if not self.test_mode:
                self.agent_caps = self.rs.choice([1, 1, 1, 2, 2, 2, 3, 3], 5, replace=False)[:, np.newaxis]
                self.prey_caps = self.rs.choice([2, 2, 2, 3, 3, 3, 4, 4], 5, replace=False)[:, np.newaxis]
                self.landmark_caps = np.zeros((self.n_landmarks, 1), dtype=np.float32)
            else:
                self.agent_caps = np.zeros((self.n_agents, 1), dtype=np.float32)
                self.prey_caps = np.zeros((self.n_preys, 1), dtype=np.float32)
                self.landmark_caps = np.zeros((self.n_landmarks, 1), dtype=np.float32)

                self.agent_caps[0, 0] = 4
                self.prey_caps[0, 0] = 5
                self.agent_caps[1:, 0] = self.rs.choice([1, 1, 2, 2, 2, 3, 3], self.n_agents - 1, replace=False)
                self.prey_caps[1:, 0] = self.rs.choice([2, 2, 3, 3, 3, 4, 4], self.n_preys - 1, replace=False)

        elif self.map_name == 'various_all':
            if not self.test_mode:
                self.n_agents = self.rs.choice([2, 3, 5, 8], 1, replace=False)[0]
                self.n_preys = self.rs.choice([2, 3, 5, 8], 1, replace=False)[0]
                self.n_landmarks = self.rs.choice([2, 4], 1, replace=False)[0]

                self.agent_caps = self.rs.choice([1, 1, 1, 2, 2, 2, 3, 3], self.n_agents, replace=False)[:, np.newaxis]
                if self.agent_caps.max() == 1:
                    self.prey_caps = self.rs.choice([2, 2, 2, 2, 2, 2, 2, 2], self.n_preys, replace=False)[:, np.newaxis]
                elif self.agent_caps.max() == 2:
                    self.prey_caps = self.rs.choice([2, 2, 2, 2, 3, 3, 3, 3], self.n_preys, replace=False)[:, np.newaxis]
                elif self.agent_caps.max() == 3:
                    self.prey_caps = self.rs.choice([2, 2, 2, 3, 3, 3, 4, 4], self.n_preys, replace=False)[:, np.newaxis]

                self.landmark_caps = np.zeros((self.n_landmarks, 1), dtype=np.float32)

            else:
                self.n_agents = self.rs.choice([4, 6, 7, 9], 1, replace=False)[0]
                self.n_preys = self.rs.choice([4, 6, 7, 9], 1, replace=False)[0]
                self.n_landmarks = self.rs.choice([3, 5], 1, replace=False)[0]

                self.agent_caps = np.zeros((self.n_agents, 1), dtype=np.float32)
                self.prey_caps = np.zeros((self.n_preys, 1), dtype=np.float32)
                self.landmark_caps = np.zeros((self.n_landmarks, 1), dtype=np.float32)

                self.agent_caps[0, 0] = 4
                self.prey_caps[0, 0] = 5
                self.agent_caps[1:, 0] = self.rs.choice([1, 1, 1, 2, 2, 2, 3, 3], self.n_agents - 1, replace=False)
                if self.agent_caps[1:, 0].max() == 1:
                    self.prey_caps[1:, 0] = self.rs.choice([2, 2, 2, 2, 2, 2, 2, 2], self.n_preys - 1, replace=False)
                elif self.agent_caps[1:, 0].max() == 2:
                    self.prey_caps[1:, 0] = self.rs.choice([2, 2, 2, 2, 3, 3, 3, 3], self.n_preys - 1, replace=False)
                elif self.agent_caps[1:, 0].max() == 3:
                    self.prey_caps[1:, 0] = self.rs.choice([2, 2, 2, 3, 3, 3, 4, 4], self.n_preys - 1, replace=False)

        self.agent_type = np.zeros((self.n_agents, 3), dtype=np.float32)
        self.prey_type = np.zeros((self.n_preys, 3), dtype=np.float32)
        self.landmark_type = np.zeros((self.n_landmarks, 3), dtype=np.float32)

        self.agent_type[:, 0] = 1
        self.prey_type[:, 1] = 1
        self.landmark_type[:, 2] = 1

        # align the id dim
        self.agent_ids = np.zeros((self.n_agents, self.max_n_agents), dtype=np.float32)
        self.prey_ids = np.zeros((self.n_preys, self.max_n_agents), dtype=np.float32)
        self.landmark_ids = np.zeros((self.n_landmarks, self.max_n_agents), dtype=np.float32)

        agent_init_ids = self.rs.choice(range(0, self.max_n_agents), self.n_agents, replace=False)
        self.agent_ids[np.arange(0, self.n_agents), agent_init_ids] = 1.0

        init_locs = self.rs.choice(range(0, self.map_size * self.map_size),  self.n_agents + self.n_preys + self.n_landmarks, replace=False)
        agent_init_locs = init_locs[:self.n_agents]
        prey_init_locs = init_locs[self.n_agents:self.n_agents + self.n_preys]
        landmark_init_locs = init_locs[self.n_agents + self.n_preys:]

        self.agent_locs = np.zeros((self.n_agents, self.map_size * self.map_size), dtype=np.float32)
        self.prey_locs = np.zeros((self.n_preys, self.map_size * self.map_size), dtype=np.float32)
        self.landmark_locs = np.zeros((self.n_landmarks, self.map_size * self.map_size), dtype=np.float32)

        self.agent_locs[np.arange(0, self.n_agents), agent_init_locs] = 1.0
        self.prey_locs[np.arange(0, self.n_preys), prey_init_locs] = 1.0
        self.landmark_locs[np.arange(0, self.n_landmarks), landmark_init_locs] = 1.0

        self.agent_coords = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.prey_coords = np.zeros((self.n_preys, 2), dtype=np.float32)
        self.landmark_coords = np.zeros((self.n_landmarks, 2), dtype=np.float32)

        self.agent_coords[np.arange(0, self.n_agents), 0] = agent_init_locs // self.map_size
        self.agent_coords[np.arange(0, self.n_agents), 1] = agent_init_locs % self.map_size
        self.prey_coords[np.arange(0, self.n_preys), 0] = prey_init_locs // self.map_size
        self.prey_coords[np.arange(0, self.n_preys), 1] = prey_init_locs % self.map_size
        self.landmark_coords[np.arange(0, self.n_landmarks), 0] = landmark_init_locs // self.map_size
        self.landmark_coords[np.arange(0, self.n_landmarks), 1] = landmark_init_locs % self.map_size

        self.map = np.zeros((self.map_size, self.map_size))
        self.map[np.array(self.agent_coords[:, 0], dtype=np.int64), np.array(self.agent_coords[:, 1], dtype=np.int64)] = np.arange(0, self.n_agents) + 1
        self.map[np.array(self.prey_coords[:, 0], dtype=np.int64), np.array(self.prey_coords[:, 1], dtype=np.int64)] = -np.arange(0, self.n_preys) - 1
        self.map[np.array(self.landmark_coords[:, 0], dtype=np.int64), np.array(self.landmark_coords[:, 1], dtype=np.int64)] = -100

        self.t = 0
        if self.entity_scheme:
            return self.get_entities(), self.get_masks()
        return self.get_obs(), self.get_state()

    def close(self):
        return

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["n_agents"] = self.n_agents
        env_info["n_fixed_actions"] = self.get_total_actions()
        env_info["n_mutual_actions"] = 0
        if self.entity_scheme:
            env_info["n_agents"] = self.max_n_agents
            env_info["n_entities"] = self.max_n_agents + self.max_n_preys + self.max_n_landmarks
            env_info["entity_shape"] = self.get_entity_size()
        return env_info
