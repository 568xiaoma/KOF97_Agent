import gym
import pygame
import matplotlib
import argparse
from gym import logger
from gymnasium import spaces
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from ppo.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from comble import *
from kof_environment import index_to_move_action_p1, index_to_move_action_p2, index_to_attack_action_p1, index_to_attack_action_p2

try:
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn("failed to set matplotlib backend, plotting will not work: %s" % str(e))
    plt = None

from copy import deepcopy
import numpy as np
from actions import Actions
from collections import deque
from pygame.locals import VIDEORESIZE
import torch
from stable_baselines3.common.utils import obs_as_tensor

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / max(arr_max - arr_min, 1e-5)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))




def index_to_move_action_p1(action):
    return {
        0: [Actions.P1_UP],
        1: [Actions.P1_RIGHT],
        2: [Actions.P1_DOWN],
        3: [Actions.P1_LEFT],
        4: [Actions.P1_UP, Actions.P1_RIGHT],
        5: [Actions.P1_RIGHT, Actions.P1_DOWN],
        6: [Actions.P1_DOWN, Actions.P1_LEFT],
        7: [Actions.P1_LEFT, Actions.P1_UP],
        8: [] # 6 3 4
    }[action]
    
def index_to_attack_action_p1(action):
    return {
        0: [Actions.P1_A],
        1: [Actions.P1_B],
        2: [Actions.P1_C],
        3: [Actions.P1_D],
        4: [Actions.P1_A, Actions.P1_B],
        5: [Actions.P1_C, Actions.P1_D],
        6: [Actions.P1_A, Actions.P1_B, Actions.P1_C],
        7: [Actions.P1_A, Actions.P1_B, Actions.P1_C, Actions.P1_D],
        8: [],
        9: [Actions.P1_D, Actions.P1_D]
    }[action]

def play(env, player=1, role="kyo", transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None, model=None):
    # env.reset()
    rendered = env.render(mode="rgb_array")
    keys_to_action = env.get_keys_to_action_p1() if player == 1 else env.get_keys_to_action_p2()
    computer = 3-player
    relevant_keys = list(set(sum(map(list, keys_to_action.keys()), []))) + ['\x10']

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size) #, flags=pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    actions_computer_candidate = []
    pownum = 0
    playing = 0
    acting = False
    comble = deepcopy(comble_base)
    comble.update(eval(f"comble_{role}"))
    agent = getattr(model, f"policy_p{computer}")
    agent(torch.zeros(1, 3, 224, 320).cuda().float())
    while running:
        if env_done:
            env_done = False
            # obs = env.reset(postfix="")[0]
            obs, playing = env.reset_display(postfix="_play")
            # if obs is not None:
            #     rendered = env.render(mode="rgb_array")
            #     display_arr(screen, rendered, transpose=transpose, video_size=video_size)
            #     pygame.display.flip()
            #     clock.tick(fps)
            actions_computer_candidate = []
            toward = 0
            wait_start = True
        elif wait_start:
            if '\x10' in pressed_keys:
                wait_start = False
                pressed_keys = []
                while playing != 32:
                    obs, playing = env.reset_display(postfix="_play")
                    if obs is not None:
                        rendered = env.render(mode="rgb_array")
                        display_arr(screen, rendered, transpose=transpose, video_size=video_size)
                        pygame.display.flip()
                        clock.tick(fps)
                # agent(torch.zeros(1, 3, 224, 320).cuda().float())
                prev_obs = obs
                obs_tensor = obs_as_tensor(obs, "cuda")
                obs_tensor = obs_tensor.permute(2, 0, 1)[None]
                obs_tensor[0]
                acting = False
        else:
            action_p2 = [keys_to_action[p] for p in pressed_keys if p != '\x10']
            prev_obs = obs
            obs_tensor = obs_as_tensor(obs, "cuda")
            obs_tensor = obs_tensor.permute(2, 0, 1)[None]
            if not acting and len(actions_computer_candidate)==0:
                actions_computer_candidate = []
                action_computer_idx, _, _ = agent(obs_tensor)
                action_computer_idx = action_computer_idx.item()
                print(action_computer_idx)
                # action_computer_idx = 9
                comble_computer = deepcopy(comble[action_computer_idx])
                if pownum > 0 and action_computer_idx == 4:
                    comble_computer = comble_computer*18
                # comble_computer = [[8, 8]]
                actions_computer_candidate.extend(comble_computer)
            if acting and len(actions_computer_candidate) == 0:
                actions_computer = [3, 8]
            else:
                actions_computer = actions_computer_candidate.pop(0)
            move_action_computer = actions_computer[0] if toward == (computer-1) else reverse[actions_computer[0]]
            actions_computer = eval(f"index_to_move_action_p{computer}")(move_action_computer)+eval(f"index_to_attack_action_p{computer}")(actions_computer[1])
            obs, rew, env_done, _, info = env.step(actions_computer+action_p2, mode="interaction")
            acting = info[f'p{computer}_acting']
            pownum = info[f'p{computer}_pownum']
            toward = info['toward']
            if callback is not None:
                callback(prev_obs, obs, action_p2, rew, env_done, info)

            if env_done:
                while info['playing'] > 160: # 168 170
                    obs, rew, env_done, _, info = env.step(actions_computer+action_p2, mode="interaction")
                    rendered = env.render(mode="rgb_array")
                    display_arr(screen, rendered, transpose=transpose, video_size=video_size)
                    pygame.display.flip()
                    clock.tick(fps)
                    pressed_keys = []
                    # for event in pygame.event.get():
                    #     # test events, set key states
                    #     if event.type == pygame.KEYDOWN:
                    #         if event.unicode == '\x10':
                    #             pressed_keys = []
                    #             break
                    print(info['playing'])

        if obs is not None:
            rendered = env.render(mode="rgb_array")
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.unicode in relevant_keys:
                    pressed_keys.append(event.unicode)
                # elif event.key == 27:
                #     running = False
            elif event.type == pygame.KEYUP:
                if event.unicode in relevant_keys:
                    if event.unicode in pressed_keys:
                        pressed_keys.remove(event.unicode)
                        
                        
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
    
    
def play_2p(env, role="kyo", transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None, model=None):
    env.reset()
    rendered = env.render(mode="rgb_array")
    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size) #, flags=pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    actions_computer_p1_candidate = []
    actions_computer_p2_candidate = []
    comble = deepcopy(comble_base)
    comble.update(eval(f"comble_{role}"))
    while running:
        if env_done:
            env_done = False
            obs = env.reset()[0]
            actions_computer_p1_candidate = []
            actions_computer_p2_candidate = []
            toward = 0
        else:
            prev_obs = obs
            obs_tensor = obs_as_tensor(obs, "cuda")
            obs_tensor = obs_tensor.permute(2, 0, 1)[None]
            
            
            if len(actions_computer_p1_candidate) == 0:
                action_computer_p1_idx, values_computer_p1, log_probs_computer_p1 = getattr(model, f"policy_p1")(obs_tensor)
                action_computer_p1_idx = action_computer_p1_idx.item()
                comble_computer_p1 = deepcopy(comble[action_computer_p1_idx])
                if action_computer_p1_idx >= 10:
                    comble_computer_p1.extend([[8, 8]]*3)
                actions_computer_p1_candidate.extend(comble_computer_p1)
            actions_computer_p1 = actions_computer_p1_candidate.pop(0)
            move_action_computer_p1 = actions_computer_p1[0] if toward == 0 else reverse[actions_computer_p1[0]]
            actions_computer_p1 = eval(f"index_to_move_action_p1")(move_action_computer_p1)+eval(f"index_to_attack_action_p1")(actions_computer_p1[1])
            
            if len(actions_computer_p2_candidate) == 0:
                action_computer_p2_idx, values_computer_p2, log_probs_computer_p2 = getattr(model, f"policy_p2")(obs_tensor)
                action_computer_p2_idx = action_computer_p2_idx.item()
                comble_computer_p2 = deepcopy(comble[action_computer_p2_idx])
                if action_computer_p2_idx >= 10:
                    comble_computer_p2.extend([[8, 8]]*3)
                actions_computer_p2_candidate.extend(comble_computer_p2)
            actions_computer_p2 = actions_computer_p2_candidate.pop(0)
            move_action_computer_p2 = actions_computer_p2[0] if toward == 0 else reverse[actions_computer_p2[0]]
            actions_computer_p2 = eval(f"index_to_move_action_p2")(move_action_computer_p2)+eval(f"index_to_attack_action_p2")(actions_computer_p2[1])
            
            
            obs, rew, env_done, _, info = env.step(actions_computer_p1+actions_computer_p2, mode="interaction")
            toward = info['toward']
            if callback is not None:
                callback(prev_obs, obs, actions_computer_p2, rew, env_done, info)
        if obs is not None:
            rendered = env.render(mode="rgb_array")
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()



class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        assert plt is not None, "matplotlib backend failed, plotting will not work"

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(
                range(xmin, xmax), list(self.data[i]), c="blue"
            )
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.000001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="MontezumaRevengeNoFrameskip-v4",
        help="Define Environment",
    )
    args = parser.parse_args()
    env = gym.make(args.env)
    play(env, zoom=4, fps=60)


if __name__ == "__main__":
    main()
