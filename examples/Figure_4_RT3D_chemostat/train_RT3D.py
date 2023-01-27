
import sys
import os

IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)
print(IMPORT_PATH)

import math
from casadi import *
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import time
from RED.agents.continuous_agents import RT3D_agent
from RED.environments.OED_env import OED_env
from RED.environments.chemostat.xdot_chemostat import xdot
import tensorflow as tf

import multiprocessing
import json


@hydra.main(version_base=None, config_path="../../RED/configs", config_name="example/RT3D_chemostat")
def train_RT3D(cfg : DictConfig):
    #setup
    cfg = cfg.example
    n_cores = multiprocessing.cpu_count()
    _, n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [cfg.environment[k] for k in cfg.environment.keys()]
    actual_params = DM(actual_params)
    normaliser = np.array(normaliser)
    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements
    param_guesses = actual_params
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)

    # agent setup
    pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, cfg.hidden_layer_size[0], cfg.hidden_layer_size[1], n_controlled_inputs]
    val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, n_observed_variables + 1 + n_controlled_inputs, cfg.hidden_layer_size[0], cfg.hidden_layer_size[1], 1]
    agent = instantiate(cfg.model, pol_layer_sizes=pol_layer_sizes, val_layer_sizes=val_layer_sizes, batch_size=int(N_control_intervals * skip))

    update_count = 0
    explore_rate = cfg.explore_rate
    all_returns = []
    all_test_returns = []

    # env setup
    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser
    env = OED_env(*args)
    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)


    for episode in range(int(n_episodes//skip)): #training loop

        actual_params = np.random.uniform(low=cfg.environment.actual_params, high=cfg.environment.actual_params, size = (skip, n_params))
        env.param_guesses = DM(actual_params)
        states = [env.get_initial_RL_state_parallel() for i in range(skip)]
        e_returns = [0 for _ in range(skip)]
        e_actions = []
        e_exploit_flags =[]
        e_rewards = [[] for _ in range(skip)]
        e_us = [[] for _ in range(skip)]
        trajectories = [[] for _ in range(skip)]
        sequences = [[[0]*pol_layer_sizes[1]] for _ in range(skip)]

        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(skip)]
        env.detFIMs = [[] for _ in range(skip)]

        for e in range(0, N_control_intervals): # run an episode
            inputs = [states, sequences]


            if episode < 1000 // skip:
                actions = agent.get_actions(inputs, explore_rate = 1, test_episode = True, recurrent=True)
            else:
                actions = agent.get_actions(inputs, explore_rate=explore_rate, test_episode=True, recurrent=True)

            e_actions.append(actions)
            outputs = env.map_parallel_step(np.array(actions).T, actual_params, continuous = True)
            next_states = []

            for i,o in enumerate(outputs): #extract outputs from experiments run in parallel
                next_state, reward, done, _, u  = o
                e_us[i].append(u)
                next_states.append(next_state)
                state = states[i]

                action = actions[i]

                if e == N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
                    done = True

                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.concatenate((state, action)))
                if reward != -1: # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward
            states = next_states

        for trajectory in trajectories: # append trajectories to memory
            if np.all( [np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
                agent.memory.append(trajectory)

        if episode > 1000 // skip: # train agent
            t = time.time()
            for _ in range(skip):
                update_count += 1
                policy = update_count % cfg.policy_delay == 0
                agent.Q_update(policy=policy, fitted=False, recurrent=True)


        explore_rate = agent.get_rate( episode, 0, 1, n_episodes / (11 * skip)) * cfg.max_std

        all_returns.extend(e_returns)

        print()
        print('EPISODE: ', episode, episode*skip)
        print('explore rate: ', explore_rate)
        print('av return: ', np.mean(all_returns[-skip:]))
        print()



    #plot and save results
    agent.save_network(save_path)
    np.save(os.path.join(save_path, 'all_returns.npy'), np.array(all_returns))
    np.save(os.path.join(save_path,'actions.npy'), np.array(agent.actions))


    t = np.arange(N_control_intervals) * int(control_interval_time)



    plt.plot(all_returns)
    plt.show()


if __name__ == '__main__':
    train_RT3D()
