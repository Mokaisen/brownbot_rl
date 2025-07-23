try:
    from isaaclab.app import AppLauncher
    import argparse

    # Launch Isaac Sim
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.headless = True
    args.enable_cameras = False
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import os
    import torch
    import torch.nn as nn
    #import gym

    import gymnasium as gym

    from skrl.models.torch import Model, GaussianMixin
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    # from isaaclab_rl.skrl.models.torch.gaussian import GaussianPolicy
    # from skrl.models.torch.gaussian import GaussianPolicy

    import brownbot_rl.tasks

    # --- [3] Define the GaussianPolicy based on your trained architecture ---
    class GaussianPolicy(Model, GaussianMixin):
        def __init__(self, observation_space, action_space, device,
                    clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
            print("[DEBUG] -> entering Model init")
            Model.__init__(self, observation_space, action_space, device)
            print("[DEBUG] -> finished Model init")
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
            print("[DEBUG] -> finished GaussianMixin init")

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

    # --------------------------------------------
    # Change these variables as needed:
    TASK_NAME = "Template-Brownbot-Rl-v0"
    ALGORITHM = "ppo"
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    EXPERIMENT_DIR = "/isaac-sim/workspaces/brownbot_rl/logs/skrl/brownbot_lift/2025-06-13_22-21-29_ppo_torch_rewardsToOne/checkpoints"
    # --------------------------------------------

    # Step 1: Load environment config and agent config
    env_cfg = parse_env_cfg(TASK_NAME, device=DEVICE, num_envs=1, use_fabric=True)
    agent_cfg = load_cfg_from_registry(TASK_NAME, f"skrl_cfg_entry_point")

    # Step 2: Build the environment
    print("[INFO] Creating environment...")
    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array")
    print("[INFO] Environment created.")
    env = SkrlVecEnvWrapper(env,ml_framework="torch")
    print("[INFO] wrap skrl env.")

    # Step 3: Reconstruct the policy model
    obs_space = env.observation_space
    act_space = env.action_space

    print("[INFO] got obs and actions from env wraper")

    print("[DEBUG] obs_space:", obs_space)
    print("[DEBUG] act_space:", act_space)
    print("[DEBUG] obs_space shape:", obs_space.shape)
    print("[DEBUG] act_space shape:", act_space.shape)
    print("[DEBUG] device:", DEVICE)

    print("[DEBUG] agent_cfg[models][policy] :", agent_cfg["models"]["policy"].get("clip_actions", False))
    policy = GaussianPolicy(
        observation_space=obs_space,
        action_space=act_space,
        device=DEVICE,
        clip_actions=agent_cfg["models"]["policy"].get("clip_actions", False),
        clip_log_std=agent_cfg["models"]["policy"].get("clip_log_std", True),
        min_log_std=agent_cfg["models"]["policy"].get("min_log_std", -20.0),
        max_log_std=agent_cfg["models"]["policy"].get("max_log_std", 2.0),
    )

    #initial_log_std=agent_cfg["models"]["policy"].get("initial_log_std", 0.0),

    # Step 4: Load the trained weights
    print(f"[INFO] Loading checkpoint: ")
    print("checkpoint path:", os.path.join(EXPERIMENT_DIR, "agent_72000.pt"))
    policy.load("/isaac-sim/workspaces/brownbot_rl/logs/skrl/brownbot_lift/2025-06-13_22-21-29_ppo_torch_rewardsToOne/checkpoints/agent_72000.pt")
    print(f"[INFO] Checkpoint loaded successfully.")

    # Step 5: Create dummy input and export as TorchScript
    dummy_input = torch.randn(1, obs_space.shape[0], device=DEVICE)
    scripted_policy = torch.jit.trace(policy.model, dummy_input)
    scripted_policy.save(os.path.join(EXPERIMENT_DIR, "policy_scripted.pt"))

    print(f"✅ TorchScript model saved to: {EXPERIMENT_DIR}/policy_scripted.pt")
finally:
    simulation_app.close()
    exit()