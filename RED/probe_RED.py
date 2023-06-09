import os
import math

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from RED.environments.probe_env import ActionObsRewardEnv, ActionRewardEnv


@hydra.main(version_base=None, config_path="configs", config_name="probe")
def run_RT3D(cfg: DictConfig):
    ### config setup
    cfg = cfg.example
    print(
        "--- Configuration ---",
        OmegaConf.to_yaml(cfg, resolve=True),
        "--- End of configuration ---",
        sep="\n\n",
    )

    ### prepare save path
    os.makedirs(cfg.save_path, exist_ok=True)
    print("Results will be saved in: ", cfg.save_path)

    ### agent setup
    agent = instantiate(cfg.model)
    explore_rate = cfg.initial_explore_rate
    seq_dim = 1

    ### env setup
    env = ActionRewardEnv()
    total_episodes = cfg.environment.n_episodes
    skip_first_n_episodes = cfg.environment.skip_first_n_experiments

    history = {k: [] for k in ["returns", "actions", "rewards", "us", "explore_rate"]}
    update_count = 0

    ### training loop
    for episode in range(total_episodes):
        ### episode buffers for agent
        # TODO: Env.reset to init()
        obs, _ = env.reset()
        states = [obs]
        trajectories = [[]]
        sequences = [[[0] * seq_dim]]

        ### episode logging buffers
        e_returns = [0]
        e_actions = []
        e_rewards = [[]]
        e_us = [[]]

        ### reset env between episodes
        env.reset()

        # Run an episode
        # TODO: Range is number of steps
        for control_interval in range(0, 1):
            inputs = [states, sequences]

            ### get agent's actions
            if episode < skip_first_n_episodes:
                actions = agent.get_actions(
                    inputs,
                    explore_rate=1,
                    test_episode=cfg.test_episode,
                    recurrent=True,
                )
            else:
                actions = agent.get_actions(
                    inputs,
                    explore_rate=explore_rate,
                    test_episode=cfg.test_episode,
                    recurrent=True,
                )
            e_actions.append(actions)

            ### step env
            next_obs, reward, done, _, _ = env.step(actions)

            transition = ()

            next_states = []
            for i, obs in enumerate(outputs):
                state, action = states[i], actions[i]
                next_state, reward, done, _, u = obs

                ### set done flag
                if (
                    control_interval == cfg.environment.N_control_intervals - 1
                    or np.all(np.abs(next_state) >= 1)
                    or math.isnan(np.sum(next_state))
                ):
                    done = True

                ### memorize transition
                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.concatenate((state, action)))

                ### log episode data
                e_us[i].append(u)
                next_states.append(next_state)
                if (
                    reward != -1
                ):  # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward
            states = next_states

        ### do not memorize the test trajectory (the last one)
        if cfg.test_episode:
            trajectories = trajectories[:-1]

        ### append trajectories to memory
        for trajectory in trajectories:
            # check for instability
            if np.all(
                [np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))]
            ) and not math.isnan(np.sum(trajectory[-1][0])):
                agent.memory.append(trajectory)

        ### train agent
        if episode > skip_first_n_episodes:
            for _ in range(cfg.environment.n_parallel_experiments):
                update_count += 1
                update_policy = update_count % cfg.policy_delay == 0
                agent.Q_update(policy=update_policy, recurrent=True)

        ### update explore rate
        explore_rate = cfg.explore_rate_mul * agent.get_rate(
            episode=episode,
            min_rate=0,
            max_rate=1,
            denominator=cfg.environment.n_episodes
            / (11 * cfg.environment.n_parallel_experiments),
        )

        ### log results
        history["returns"].extend(e_returns)
        history["actions"].extend(np.array(e_actions).transpose(1, 0, 2))
        history["rewards"].extend(e_rewards)
        history["us"].extend(e_us)
        history["explore_rate"].append(explore_rate)

        print(
            f"\nEPISODE: [{episode}/{total_episodes}] ({episode * cfg.environment.n_parallel_experiments} experiments)",
            f"explore rate:\t{explore_rate:.2f}",
            f"average return:\t{np.mean(e_returns):.5f}",
            sep="\n",
        )


if __name__ == "__main__":
    run_RT3D()
