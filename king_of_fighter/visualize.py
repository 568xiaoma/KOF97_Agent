from kof_environment import Environment
from monitor import Monitor
from king_of_fight_custom_wrapper import KoFCustomWrapper
import random
from copy import deepcopy
import numpy as np
from keyboard_play import play, play_2p
from ppo.ppo import PPO
import os
from baseline import ActorCriticCnnPolicy, lr_schedule, clip_range_schedule

comble_base = {
    0: [[8, 4]], # 闪躲
    1: [[8, 6]], # 爆气
    2: [[3, 8], [3, 8], [3, 8]], # 防御
    3: [[6, 8], [6, 8], [6, 8]], # 蹲防
}
comble_kyo = {
    4: [[5, 3]],
    5: [[1, 8], [2, 8], [5, 2]],
    6: [[3, 8], [2, 8], [6, 3]],
    7: [[2, 8], [6, 8], [3, 2]],
    8: [[2, 8], [5, 8], [1, 0]],
    9:[[2, 8], [5, 8], [1, 0], 
        [2, 8], [5, 8], [1, 2]],
    10:[[2, 8], [5, 8], [1, 0], 
        [2, 8], [5, 8], [1, 2], 
        [8, 8], [8, 8], [8, 1]],
    11:[[2, 8], [5, 8], [1, 0], 
        [2, 8], [5, 8], [1, 2], 
        [8, 8], [8, 8], [8, 0]],
    12:[[2, 8], [5, 8], [1, 2]],
    13:[[2, 8], [6, 8], [3, 8], [6, 8], [2, 8], [5, 8], [1, 0]],
    14:[[2, 8], [5, 8], [1, 8], [2, 8], [5, 8], [1, 0]],
}

reverse = {0:0, 1:3, 2:2, 3:1, 4:7, 5:6, 6:5, 7:4, 8:8}


def make_env(seed=0, rank=0):
    def _init(role='mai'):
        roms_path = "roms/"
        env = Environment("env1", roms_path, frames_per_step=1, frame_ratio=1, render=False)
        env.seed(seed+rank)
        env = KoFCustomWrapper(env, role=role)
        env = Monitor(env, role=role)
        return env
    return _init


role = 'kyo'
roms_path = "roms/"  # Replace this with the path to your ROMs
os.system("Xvfb :0 -screen 0 800x600x16 +extension RANDR &")
os.environ['DISPLAY'] = ":0"
env = make_env(0)(role=role)
# model = PPO.load(f"trained_models/ppo_{role}_div_2p_3000000_steps.zip", env=env)
model = PPO(
    ActorCriticCnnPolicy, #"CnnPolicy", 
    env,
    device="cuda", 
    verbose=1,
    n_steps=512,
    batch_size=512,
    n_epochs=4,
    gamma=0.94,
    learning_rate=lr_schedule,
    clip_range=clip_range_schedule,
    tensorboard_log=f"tensorboard/{role}_logs"
)
env = KoFCustomWrapper(Environment("env1", "roms/", frames_per_step=1, frame_ratio=3), role=role)
os.environ['DISPLAY'] = "localhost:10.0"
play_2p(env, model=model, role=role)