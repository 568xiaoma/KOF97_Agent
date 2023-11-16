# Copyright 2023 LIN Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import time
import collections

import cv2
import gym
import random
import gymnasium
import numpy as np
from copy import deepcopy
from comble import *

# Custom environment wrapper
class KoFCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False, role="mai", num_actions=11):
        super(KoFCustomWrapper, self).__init__(env)
        self.env = env
        self.project_dir = env.project_dir
        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)
        self.num_step_frames = 1
        self.reward_coeff = 3.0
        self.total_timesteps = 0

        self.full_hp = 103
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.num_actions = num_actions
        self.action_space = gymnasium.spaces.discrete.Discrete(self.num_actions)
        self.observation_space = gymnasium.spaces.box.Box(low=0, high=255, shape=(112, 160, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering
        self.toward = 0
        self.actions_p1_candidate = []
        self.actions_p2_candidate = []
        self.role = role
        self.prev_pownum_p1 = 0
        self.prev_pownum_p2 = 0
        self.comble = deepcopy(comble_base)
        self.comble.update(eval(f"comble_{role}"))
    
    def _stack_observation(self):
        return np.stack([cv2.cvtColor(self.frame_stack[i * 3 + 2], cv2.COLOR_RGB2GRAY) for i in range(3)], axis=-1)

    def render(self, mode="rgb_array"):
        return self.env.render(mode)
    
    def reset_display(self, postfix="", **kwargs):
        observation, playing = self.env.reset_display(status=f"{self.project_dir}/status/{self.role}_2p_started{postfix}")
        if playing == 32:
            self.prev_player_health = self.full_hp
            self.prev_oppont_health = self.full_hp
            self.prev_combo_p1 = 0
            self.prev_combo_p2 = 0
            self.prev_pownum_p1 = 0
            self.prev_pownum_p2 = 0
            self.total_timesteps = 0
            self.frame_stack.clear()
            for _ in range(self.num_frames):
                self.frame_stack.append(observation[::2, ::2, :])
            self.reward_total = 0
        return observation, playing


    def reset(self, postfix="", **kwargs):
        observation = self.env.reset(status=f"{self.project_dir}/status/{self.role}_2p_started{postfix}")
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp
        self.prev_combo_p1 = 0
        self.prev_combo_p2 = 0
        self.prev_pownum_p1 = 0
        self.prev_pownum_p2 = 0
        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        self.reward_total = 0
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1), None

    def flip(self, action):
        new_action = []
        for move_action, attack_action in action:
            new_action.append([reverse[move_action], attack_action])
        return new_action

    def step(self, action, mode='computer_2p'):
        if mode == 'computer_2p':
            action_p1_idx, action_p2_idx = action
            if len(self.actions_p1_candidate) == 0 and action_p1_idx != -1:
                actions_p1 = deepcopy(self.comble[action_p1_idx])
                if action_p1_idx == 4:
                    if self.prev_pownum_p1 == 0:
                        actions_p1 = [[3, 8]]
                    else:
                        actions_p1 = actions_p1*18
                self.actions_p1_candidate.extend(actions_p1)
            if len(self.actions_p2_candidate) == 0 and action_p2_idx != -1:
                actions_p2 = deepcopy(self.comble[action_p2_idx])
                if action_p2_idx == 4:
                    if self.prev_pownum_p2 == 0:
                        actions_p2 = [[3, 8]]
                    else:
                        actions_p2 = actions_p2*18
                self.actions_p2_candidate.extend(self.flip(actions_p2))
            if len(self.actions_p1_candidate) == 0 and action_p1_idx == -1:
                self.actions_p1_candidate.append([3, 8])
            if len(self.actions_p2_candidate) == 0 and action_p2_idx == -1:
                self.actions_p2_candidate.append([3, 8])

            move_action_p1, attack_action_p1 = self.actions_p1_candidate.pop(0)
            move_action_p2, attack_action_p2 = self.actions_p2_candidate.pop(0)
            if self.toward == 1:
                move_action_p1 = reverse[move_action_p1]
                move_action_p2 = reverse[move_action_p2]
            action = [move_action_p1, move_action_p2, attack_action_p1, attack_action_p2]
            obs, _reward, _done, info = self.env.step(action, mode=mode)
            self.toward = info['toward']
            self.frame_stack.append(obs[::2, ::2, :])
            if len(self.frame_stack) > self.num_frames:
                self.num_frames.pop(0)
            info['p1_next_step'] = len(self.actions_p1_candidate) == 0 and not info['p1_acting']
            info['p2_next_step'] = len(self.actions_p2_candidate) == 0 and not info['p2_acting']
            info['action_p1'] = [move_action_p1, attack_action_p1]
            info['action_p2'] = [move_action_p2, attack_action_p2]
            custom_reward = self.reward(info)
            self.prev_pownum_p1 = info['p1_pownum']
            self.prev_pownum_p2 = info['p2_pownum']
            return self._stack_observation(), custom_reward, _done, _done, info # reward normalization
        
        else:
            obs, _reward, _done, info = self.env.step(action, mode=mode)
            self.frame_stack.append(obs[::2, ::2, :])
            if len(self.frame_stack) > self.num_frames:
                self.num_frames.pop(0)
            return self._stack_observation(), None, _done, _done, info
            
            
    def reward(self, info):
        curr_player_health = info['healthP1']
        curr_oppont_health = info['healthP2']
        
        self.total_timesteps += self.num_step_frames
        
        if info['win'] == 'FALSE':
            damage_reward_p1 = -self.full_hp-max(curr_oppont_health, 0)
            damage_reward_p2 =  self.full_hp+max(curr_oppont_health, 0)
        elif info['win'] == 'TRUE':
            damage_reward_p1 =  self.full_hp+max(curr_player_health, 0)
            damage_reward_p2 = -self.full_hp-max(curr_player_health, 0)
        else:
            oppont_diff = self.prev_oppont_health - curr_oppont_health
            player_diff = self.prev_player_health - curr_player_health
            damage_reward_p1 = (oppont_diff - player_diff)
            damage_reward_p2 = (player_diff - oppont_diff)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            
        distance = np.abs(info["1P_x"] - info["2P_x"])
        if distance <= 150:
            distance_reward = 0
        else:
            distance_reward = -(distance-150)
        # combo_reward_p1 = (info['2P_combo']-self.prev_combo_p2)-(info['1P_combo']-self.prev_combo_p1)
        # combo_reward_p2 = (info['1P_combo']-self.prev_combo_p1)-(info['2P_combo']-self.prev_combo_p2)
        # self.prev_combo_p1 = info['1P_combo']
        # self.prev_combo_p2 = info['2P_combo']
        custom_reward_p1 = 0.01*damage_reward_p1 # +0.0001*distance_reward + 0.05*combo_reward_p1
        custom_reward_p2 = 0.01*damage_reward_p2 # +0.0001*distance_reward + 0.05*combo_reward_p2
        info['damage_reward_p1'] = damage_reward_p1
        info['damage_reward_p2'] = damage_reward_p2
        info['distance_reward'] = distance_reward
        # info['combo_reward_p1'] = combo_reward_p1
        # info['combo_reward_p2'] = combo_reward_p2
        return (custom_reward_p1, custom_reward_p2)
        
        
    def print_log(self, data, comble):
        print("stage: %02d, done: %1d, win: %1d/%1d, reward: %4.3f, time: %03d, playing: %03d, hp1: %.3d, hp2: %03d, x1: %03d, x2: %03d, comble: %02d"%(
            self.env.stage, data['done'], self.env.status.count('TRUE'), 
            self.env.status.count('FALSE'), self.reward_total, 
            data['time'], data['playing'], 
            data['healthP1'], data['healthP2'],
            data['1P_x'], data['2P_x'],
            comble))