from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
from gymnasium.utils.save_video import save_video

env = gym.make("Hopper-v4", render_mode='rgb_array_list')





