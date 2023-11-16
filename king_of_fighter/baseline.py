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

import os
import sys

import retro
# from stable_baselines3 import PPO
from ppo.ppo import PPO
from monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from kof_environment import Environment
from king_of_fight_custom_wrapper import KoFCustomWrapper
from ppo.cnn_policy import ActorCriticCnnPolicy
from ppo.resnet_policy import ActorCriticResnetPolicy
import argparse

os.system("Xvfb :0 -screen 0 800x600x16 +extension RANDR &")
os.environ["DISPLAY"] = ":0"
# os.environ['DISPLAY'] = "node-f7954-4:27.0"

NUM_ENV = 16
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(seed=0, rank=0):
    def _init(role='mai', project_dir='', num_actions=11):
        roms_path = "roms/"
        env = Environment("env1", roms_path, frames_per_step=1, frame_ratio=2, project_dir=project_dir)
        env.seed(seed+rank)
        env = KoFCustomWrapper(env, role=role, num_actions=num_actions)
        env = Monitor(env, role=role)
        return env
    return _init

lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
clip_range_schedule = linear_schedule(0.15, 0.025)

def main():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--role', default="kyo", type=str,help ='input files') 
    args = parser.parse_args()
    role = args.role
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = make_env(0)(role=role, project_dir=project_dir, num_actions=11)
    # env = Environment("env1", "roms/", frames_per_step=1, frame_ratio=3)
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    model = PPO(
        ActorCriticResnetPolicy, #"CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.95,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=f"tensorboard/{role}_logs"
    )
    # model = PPO.load(f"trained_models/ppo_{role}_resnet_2p_50000_steps.zip", env=env)
    # model = PPO.load(f"trained_models/ppo_kyo_div_2p_100000_steps.zip", env=env)
    # Set the save directory
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_interval = 5000
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix=f"ppo_{role}_resnet_2p")

    # Writing the training logs from stdout to a file
    model.learn(
        tb_log_name=role,
        total_timesteps=int(100000000), # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
        callback=[checkpoint_callback]#, stage_increase_callback]
    )
    env.close()
    # Save the final model
    model.save(os.path.join(save_dir, "ppo_{role}_resnet_2p_final.zip"))

if __name__ == "__main__":
    main()
