from copy import deepcopy
from typing import Tuple, Callable, List

import torch
import numpy as np
from tqdm import tqdm

import Box2D
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium.envs.box2d.bipedal_walker import (
    FPS,
    SCALE,
    VIEWPORT_H,
    VIEWPORT_W,
    SPEED_HIP,
    SPEED_KNEE,
    LEG_H,
    LEG_DOWN
)

from .base_dataset import BasePreferenceDataset


class BipedalWalkerResetWrapper(gym.Wrapper):
    """
    A wrapper class for gym BipedalWalker-v3
    that enables environment reset with any given initial state.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env.unwrapped

    def reset(self, state=None, **kwargs):
        # Call the original reset to initialize everything
        obs, info = self.env.reset(**kwargs)

        if state is not None:
            # Set the hull's initial state
            self.env.hull.angle = state[0]
            self.env.hull.angularVelocity = state[1] * FPS / 2.0
            vel_x = state[2] * (SCALE / VIEWPORT_W) * FPS / 0.3
            vel_y = state[3] * (SCALE / VIEWPORT_H) * FPS / 0.3
            self.env.hull.linearVelocity = Box2D.b2Vec2(vel_x, vel_y)

            # Set leg positions to approximate joint angles
            left_leg_pos = Box2D.b2Vec2(
                self.env.hull.position.x,
                self.env.hull.position.y - LEG_H / 2 - LEG_DOWN,
            )
            right_leg_pos = Box2D.b2Vec2(
                self.env.hull.position.x,
                self.env.hull.position.y - LEG_H / 2 - LEG_DOWN,
            )

            # Set positions and angles for left leg
            self.env.legs[0].position = left_leg_pos
            self.env.legs[0].angle = state[4]
            self.env.joints[0].motorSpeed = state[5] * SPEED_HIP
            self.env.joints[1].motorSpeed = state[7] * SPEED_KNEE
            self.env.legs[1].ground_contact = bool(state[8])

            # Set positions and angles for right leg
            self.env.legs[2].position = right_leg_pos
            self.env.legs[2].angle = state[9]
            self.env.joints[2].motorSpeed = state[10] * SPEED_HIP
            self.env.joints[3].motorSpeed = state[12] * SPEED_KNEE
            self.env.legs[3].ground_contact = bool(state[13])

            # Override the observation to match the new state
            obs = np.array(state, dtype=np.float32)

        return obs, info


def get_discount_returns(rewards, discount=1):
    returns = 0
    scale = 1
    for r in rewards:
        returns += scale * r
        scale *= discount
    return returns


def generate_trajectory(
    env: gym.Env, 
    policy: BaseAlgorithm,
    reward_function: Callable,
    start_state: np.ndarray=None,
    traj_len: int=20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a trajectory given an environment and a policy.
    """
    _, _ = env.reset(state=start_state)

    done = False
    state = np.copy(start_state)
    states, actions, rewards = [], [], []
    for _ in range(traj_len):
        with torch.no_grad():
            action, _ = policy.predict(state, deterministic=False)
        
        next_state, _, terminated, truncated, _ = env.step(action)
        reward = reward_function(next_state)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state   
        if (terminated or truncated):
            break

    return (
        np.array(states), 
        np.array(actions), 
        np.array(rewards), 
        np.array(start_state),
    )


def generate_pbrl_dataset(
    env: gym.Env, 
    policy: BaseAlgorithm, 
    reward_function: Callable,
    num_pairs: int=1000, 
    seg_len: int=30,
    max_trials: int=1e6,
    min_start_state: int=100,
    max_start_state: int=200,
    ) -> torch.utils.data.Dataset:
    """
    Generates a preference dataset for a given environment, 
    where the pairs are actions sequences sampled from trajectories with the same initial state.
    """
    contexts = []
    positive = []
    negative = []
    for _ in tqdm(range(num_pairs), desc='Generating Preference'):
        _sucess = False
        while not _sucess:
            
            try:
                random_start = np.random.randint(min_start_state, max_start_state)
                start_state, _ = env.reset()
                for _ in range(random_start):
                    with torch.no_grad():
                        action, _ = policy.predict(start_state, deterministic=False)
                    
                    start_state, _, terminated, truncated, _ = env.step(action)
                    assert not (terminated or truncated), 'Random start state is terminal'
                
                start_state = start_state.astype(np.float64)
                traj_0 = generate_trajectory(deepcopy(env), policy, reward_function, start_state)
                traj_1 = generate_trajectory(deepcopy(env), policy, reward_function, start_state)
                
                s0, a0, r0, x0 = traj_0
                s1, a1, r1, x1 = traj_1
                
                assert len(s0) >= seg_len, f'Trajectory length {s0.shape[0]} is less than seg_len'
                assert len(s1) >= seg_len, f'Trajectory length {s1.shape[0]} is less than seg_len'
                assert (x0 == x1).all(), 'Initial states are different'
                
                g0 = get_discount_returns(r0, discount=1)
                g1 = get_discount_returns(r1, discount=1)
                if g1 < g0:
                    s1, s0 = s0[:seg_len], s1[:seg_len]
                    a1, a0 = a0[:seg_len], a1[:seg_len]
                else:
                    s0, s1 = s1[:seg_len], s0[:seg_len]
                    a0, a1 = a1[:seg_len], a0[:seg_len]
                
                positive.append(torch.Tensor(a1.flatten()))
                negative.append(torch.Tensor(a0.flatten()))
                contexts.append(torch.Tensor(x0.flatten()))
                _sucess = True
                
            except AssertionError as e:
                max_trials -= 1
            
            if max_trials == 0:
                break

    return BasePreferenceDataset(positive, negative, contexts)
