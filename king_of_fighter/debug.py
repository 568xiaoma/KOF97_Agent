from kof_environment import Environment
from monitor import Monitor
from king_of_fight_custom_wrapper import KoFCustomWrapper
import random
from copy import deepcopy
import numpy as np
from keyboard_play import play
from ppo.ppo import PPO
import os
from ppo.resnet_policy import ActorCriticResnetPolicy
from baseline import lr_schedule, clip_range_schedule
import argparse

def make_env(seed=0, rank=0):
    def _init(role='mai', project_dir="", num_actions=10):
        roms_path = "roms/"
        env = Environment("env1", roms_path, frames_per_step=1, 
                          frame_ratio=1, render=False, project_dir=project_dir)
        env.seed(seed+rank)
        env = KoFCustomWrapper(env, role=role, num_actions=num_actions)
        env = Monitor(env, role=role)
        return env
    return _init


parser = argparse.ArgumentParser(description='play')
parser.add_argument('--role', default="kyo", type=str, choices=["kyo", "iori", "mai"], help ='role')
parser.add_argument('--player', default=2, type=int, choices=[1, 2], help ='player') 
parser.add_argument('--display', default="node-f7954-4:16.0", type=str, help ='display')
args = parser.parse_args()
role = args.role
player = args.player
display = args.display

num_actions = 11
roms_path = "roms/"  # Replace this with the path to your ROMs
os.system("Xvfb :0 -screen 0 800x600x16 +extension RANDR &")
os.environ['DISPLAY'] = ":0"
# os.environ['DISPLAY'] = display
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env = make_env(0)(role=role, project_dir=project_dir, num_actions=num_actions)

# model = PPO(
#     ActorCriticResnetPolicy, #"CnnPolicy", 
#     env,
#     device="cuda", 
#     verbose=1,
#     n_steps=512,
#     batch_size=512,
#     n_epochs=4,
#     gamma=0.95,
#     learning_rate=lr_schedule,
#     clip_range=clip_range_schedule,
#     tensorboard_log=f"tensorboard/{role}_logs"
# )
model = PPO.load(f"trained_models/ppo_{role}_resnet_2p_100000_steps.zip", env=env)
env = KoFCustomWrapper(Environment("env1", "roms/", 
                                   frames_per_step=1, 
                                   frame_ratio=2, 
                                   project_dir=project_dir), 
                       role=role, 
                       num_actions=num_actions)
os.environ['DISPLAY'] = display
play(env, model=model, player=player, role=role)