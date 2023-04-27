
import os
import sys

IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(IMPORT_PATH)


import datetime
import json
import math
import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from casadi import *

from RED.agents.continuous_agents import RT3D_agent
from RED.environments.chemostat.xdot_chemostat import xdot
from RED.environments.OED_env import OED_env

if __name__ == '__main__':
    #setup
    n_cores = multiprocessing.cpu_count()
    param_dir = os.path.join(os.path.join(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'RED',
                     'environments'), 'chemostat'))
    params = json.load(open(os.path.join(param_dir, 'params_chemostat.json')))
    n_episodes, skip, y0, actual_params, input_bounds, n_controlled_inputs, num_inputs, dt, lb, ub, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]
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
    save_path = os.path.join('.', 'results', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_path)
    print(f"[INFO] Saving to {save_path}")


    #agent setup
    pol_learning_rate = 0.00005
    hidden_layer_size = [[64, 64], [128, 128]]
    pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, hidden_layer_size[0], hidden_layer_size[1], n_controlled_inputs]
    val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, n_observed_variables + 1 + n_controlled_inputs, hidden_layer_size[0], hidden_layer_size[1], 1]
    agent = RT3D_agent(val_layer_sizes = val_layer_sizes, pol_layer_sizes = pol_layer_sizes,  policy_act = tf.nn.sigmoid, val_learning_rate = 0.0001, pol_learning_rate = pol_learning_rate)#, pol_learning_rate=0.0001)
    agent.batch_size = int(N_control_intervals * skip)
    agent.max_length = 11
    agent.mem_size = 500000000
    agent.std = 0.1
    agent.noise_bounds = [-0.25, 0.25]
    agent.action_bounds = [0, 1]
    policy_delay = 2
    update_count = 0
    max_std = 1  # for exploring
    explore_rate = max_std
    alpha = 1
    all_returns = []
    all_test_returns = []

    ### Manual loading of checkpoints
    # agent.load_network("./results/2023-03-24_13-18")

    #env setup
    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser
    env = OED_env(*args)
    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)

    history = {k: [] for k in ["returns", "actions", "rewards", "us", "explore_rate"]}

    for episode in range(int(n_episodes//skip)): #training loop
        actual_params = np.random.uniform(low=lb, high=ub,  size = (skip, 3)) # sample from uniform distribution
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

            for i,o in enumerate(outputs): #extract outputs from parallel experiments
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
                else:
                    e_rewards[i].append(0)
                    e_returns[i] += 0
            states = next_states

        for trajectory in trajectories:
            if np.all( [np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))] ) and not math.isnan(np.sum(trajectory[-1][0])): # check for instability
                agent.memory.append(trajectory)


        if episode > 1000 // skip: #train agent
            print('training', update_count)
            t = time.time()
            for _ in range(skip):

                update_count += 1
                policy = update_count % policy_delay == 0

                agent.Q_update(policy=policy, fitted=False, recurrent=True)
            print('fitting time', time.time() - t)

        explore_rate = agent.get_rate(episode, 0, 1, n_episodes / (11 * skip)) * max_std


        # all_returns.extend(e_returns)
        # print()
        # print('EPISODE: ', episode, episode*skip)
        # print('av return: ', np.mean(all_returns[-skip:]))
        # print()

        ### log results
        history["returns"].extend(e_returns)
        history["actions"].extend(np.array(e_actions).transpose(1, 0, 2))
        history["rewards"].extend(e_rewards)
        history["us"].extend(e_us)
        history["explore_rate"].append(explore_rate)

        print(
            f"\nEPISODE: [{episode}/{int(n_episodes//skip)}] ({episode * skip} experiments)",
            f"explore rate:\t{explore_rate:.2f}",
            f"average return:\t{np.mean(history['returns'][-skip:]):.5f}",
            sep="\n",
        )

        ### checkpoint
        if episode % 50 == 0:
            ckpt_dir = os.path.join(save_path, f"ckpt_{episode}")
            os.makedirs(ckpt_dir, exist_ok=True)
            agent.save_network(ckpt_dir)
            for k in history.keys():
                np.save(os.path.join(ckpt_dir, f"{k}.npy"), np.array(history[k]))

    ### save results
    agent.save_network(save_path)
    for k in history.keys():
        np.save(os.path.join(save_path, f"{k}.npy"), np.array(history[k]))
    
    #plot and save results
    # agent.save_network(save_path)
    # np.save(os.path.join(save_path, 'all_returns.npy'), np.array(all_returns))
    # np.save(os.path.join(save_path, 'actions.npy'), np.array(agent.actions))
    
    # t = np.arange(N_control_intervals) * int(control_interval_time)
    # plt.plot(all_returns)
    # plt.show()

