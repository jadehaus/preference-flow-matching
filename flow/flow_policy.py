from copy import deepcopy

import gym
import torch
import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm

from .flow_matching import OptimalTransportConditionalFlowMatching


class FlowPolicy:
    def __init__(
        self, 
        env: gym.Env, 
        flow: OptimalTransportConditionalFlowMatching, 
        policy: BaseAlgorithm, 
        seg_len: int=30, 
        iteration: int=1,
        rollout_frequency: int=1,
        device: torch.device='cuda'
        ) -> None:
        
        self.env = deepcopy(env)
        self.flow = flow
        self.policy = policy
        self.seg_len = seg_len
        self.device = device
        
        self.iteration = iteration
        self.rollout_frequency = rollout_frequency
        self.current_rollout = rollout_frequency
        self.stored_actions = None
        
    def __call__(
        self, 
        start_state: np.ndarray, 
        **kwargs
        ) -> np.ndarray:
        
        if self.current_rollout >= self.rollout_frequency:
            start_state = start_state.astype(np.float64)
            _, _ = self.env.reset(state=start_state)
            
            done = False
            actions = []
            state = np.copy(start_state)
            for _ in range(self.seg_len):
                if not done:
                    with torch.no_grad():
                        action, _ = self.policy.predict(state, deterministic=False)
                    state, _, terminated, truncated, _ = self.env.step(action)
                    done = (terminated or truncated)
                    actions.append(action)
                else:
                    actions.append(np.zeros(self.env.action_space.shape))
            
            start_state = torch.tensor(start_state).float().to(self.device)
            actions = torch.tensor(np.array(actions)).float().flatten().to(self.device)
            
            for _ in range(self.iteration):
                actions = self.flow.compute_target(actions, context=start_state, **kwargs)
            actions = actions.view(-1, self.env.action_space.shape[0])
            self.stored_actions = actions
            self.current_rollout = 0
        
        action = self.stored_actions[self.current_rollout, :]
        self.current_rollout += 1
        return action.cpu().detach().numpy()
    
    def to(self, device):
        self.device = device
        self.policy.to(device)
        return self