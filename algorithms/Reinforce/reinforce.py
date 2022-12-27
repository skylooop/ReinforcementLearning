from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import os

import typing as tp
import pyrallis
from absl import app, flags
from policy import Reinforce

import gymnasium as gym
from gymnasium.utils.save_video import save_video

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", default=42, help="Define seed for run.")
flags.DEFINE_string("env_name", default="InvertedPendulum-v4", help="Env name.")
flags.DEFINE_string("render_mode", default='rgb_array_list', help="Mode for rendering env.")
flags.DEFINE_float("num_episodes", default=5e3, help="Number of episodes to run for.")

# Hyperparameters for environment
flags.DEFINE_float("lr", default=3e-4, help="Learning rate.")
flags.DEFINE_float("gamma", default=0.99, help="Discount factor.")
flags.DEFINE_float("eps", default=1e-6, help="Numerical stability.")

#Simple two layer net with those number of neurons
flags.DEFINE_integer("hidden_size_first", default=16, help="Number of neurons in first layer.")
flags.DEFINE_integer("hidden_size_second", default=32, help="Number of neurons in second layer.")


def main(_):
    env = gym.make(FLAGS.env_name, render_mode=FLAGS.render_mode)
    # Get cumulative reward for last 50 steps
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
    
    # Obs size for state
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    rewards_over_episodes = []
    agent = Reinforce(obs_size, act_size)
    step_index = 0
    
    for episode in range(int(FLAGS.num_episodes)):
        obs, info = wrapped_env.reset()
        done = False
        
        while not done:
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            
            done = terminated or truncated
            '''
            if done:
                gym.utils.save_video.save_video(
                    wrapped_env.render(),
                    "/home/m_bobrin/ReinforcementLearning/assets",
                    fps=env.metadata['render_fps'],
                    episode_index = episode,
                    video_length = 30
                )'''
        #rewards_over_episodes.append(wrapped_env.return_queue)
        agent.update()
        
        if episode % 500 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

        
if __name__ == "__main__":
    app.run(main)
        
