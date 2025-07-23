import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Box
from skrl.models.torch import Model, GaussianMixin

class GaussianPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device="cpu",
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        print("[DEBUG] entering GaussianPolicy init")
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

#obs_space = Box(-float("inf"), float("inf"), shape=(45,), dtype=torch.float32)
#act_space = Box(-float("inf"), float("inf"), shape=(7,), dtype=torch.float32)

obs_space = Box(-np.inf, np.inf, shape=(45,), dtype=np.float32)
act_space = Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

print("[DEBUG] creating policy")
policy = GaussianPolicy(obs_space, act_space)
print("[DEBUG] policy created")