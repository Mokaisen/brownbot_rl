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

    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils.runner.torch import Runner
    from typing import Tuple

    import brownbot_rl.tasks
    
    class ScriptedGaussianPolicy(nn.Module):
        def __init__(self, preprocessor, trained_policy):
            super().__init__()
            self.net = trained_policy.net_container
            self.policy_layer = trained_policy.policy_layer

            # Clone the trained log_std parameter
            self.log_std = nn.Parameter(
                trained_policy.state_dict()["log_std_parameter"].clone().detach()
            )

            self.preprocessor = preprocessor

        def forward(self, states: torch.Tensor) -> torch.Tensor:
            states_preprocessed = self.preprocessor(states)
            x = self.net(states_preprocessed)
            mean = self.policy_layer(x)

            return mean

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

    # create runner to compute some actions given some observations taken from play.py script of skrl
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, agent_cfg)
    print("[INFO] Runner created.")

    #testing the checkpoint file
    ckpt_path = os.path.join(EXPERIMENT_DIR, "agent_72000.pt")

    print(f"[INFO] Loading model checkpoint from: {ckpt_path}")
    runner.agent.load(ckpt_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()

    print("[INFO] Environment reset.")
    print("[Debug] obs reset: ")
    print(obs)

    policy_runner = runner.agent.policy  # ← This should be the trained one
    print("[INFO] policy_runner created.")

    state_preprocessor = runner.agent._state_preprocessor
    print("[INFO] state_preprocessor: ", state_preprocessor)
    obs_preprocess = state_preprocessor(obs)

    policy_runner.eval()
    with torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        print("[INFO] outputs computed.")
        skrl_action = outputs[-1].get("mean_actions", outputs[0])  # if single-agent
        print("[INFO] skrl action: ", skrl_action)

        # 1. Use policy.act() to get actions properly
        actions, log_prob, outputs = runner.agent.policy.act({"states": obs}, role="policy")
        print("agent.actions: ", actions)
        print("agent.outputs: ", outputs)

        mean, log_std, _ = policy_runner.compute({"states": obs}, role="policy")
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        #action_mean = dist.sample()  # or dist.sample()
        action_mean = mean
        print("[INFO] action dist.mean: ", action_mean)

    print("policy_runner.net_container:")
    print(policy_runner.net_container)
    print("[INFO] scripted policy_runner.net")

    print("[INFO] policy runner modules: -------")
    print(dict(policy_runner.named_modules()).keys())
    print("[INFO] policy runner params: ")
    for name, param in policy_runner.named_parameters():
        print(f"{name}: {param.shape}")

    exportable_policy = ScriptedGaussianPolicy(state_preprocessor, policy_runner)
    print("[INFO] create exportable policy")

    exportable_policy.eval()
    with torch.no_grad():
       scripted = torch.jit.script(exportable_policy)
       scripted.save("policy_scripted.pt")
    
    # Print a sample weight from export_model to verify it changed
    print("[INFO] sample weight from exportable_policy: ")
    print(exportable_policy.net[0].weight.data.mean())  # Check if it's NOT close to 0

    # Reload TorchScript policy for testing
    print("[INFO] Reloading TorchScript policy for testing...")
    loaded_scripted_policy = torch.jit.load("policy_scripted.pt").to(DEVICE)

    # Compare actions
    print("obs.shape: ", obs.shape)
    print("obs for testing of reloaded policy: ", obs)
    with torch.inference_mode():
        mean = loaded_scripted_policy(obs)
        scripted_action = mean

    # Print and compare
    print("Original SKRL action:", skrl_action)
    print("TorchScript action:   ", scripted_action)
    print("[INFO] action dist.mean: ", action_mean)

    # compare architectures between original and exportable policy
    print("Original SKRL policy architecture:\n", policy_runner)
    print("Exportable policy architecture:\n", exportable_policy)

    #compare weights between original and exportable policy
    #the values to print below should be zero or very close to zero``
    for name1, param1 in policy_runner.named_parameters():
        if "value_layer" in name1 or "log_std" in name1:
            continue
        name2 = name1.replace("net_container", "net")
        if name2 in exportable_policy.state_dict():
            param2 = exportable_policy.state_dict()[name2]
            diff = (param1 - param2).abs().mean()
            print(f"🔍 {name1} vs {name2} | Mean abs diff: {diff:.6f}")
        else:
            print(f"⚠️ No match found for {name1}")

finally:
    simulation_app.close()
    exit()