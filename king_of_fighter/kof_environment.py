from MAMEToolkit.emulator import Emulator
from MAMEToolkit.emulator import Address
import random
from steps import *
# from MAMEToolkit.sf_environment.Actions import Actions
from actions import Actions
from address import setup_memory_addresses
import gym
import glob
import torch
import numpy as np
import random
import os

# Combines the data of multiple time steps
def add_rewards(old_data, new_data):
    for k in old_data.keys():
        if "rewards" in k:
            for player in old_data[k]:
                new_data[k][player] += old_data[k][player]
    return new_data


# Converts and index (action) into the relevant movement action Enum, depending on the player
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
    
def index_to_move_action_p2(action):
    return {
        0: [Actions.P2_UP],
        1: [Actions.P2_RIGHT],
        2: [Actions.P2_DOWN],
        3: [Actions.P2_LEFT],
        4: [Actions.P2_UP, Actions.P2_RIGHT],
        5: [Actions.P2_RIGHT, Actions.P2_DOWN],
        6: [Actions.P2_DOWN, Actions.P2_LEFT],
        7: [Actions.P2_LEFT, Actions.P2_UP],
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

def index_to_attack_action_p2(action):
    return {
        0: [Actions.P2_A],
        1: [Actions.P2_B],
        2: [Actions.P2_C],
        3: [Actions.P2_D],
        4: [Actions.P2_A, Actions.P2_B],
        5: [Actions.P2_C, Actions.P2_D],
        6: [Actions.P2_A, Actions.P2_B, Actions.P2_C],
        7: [Actions.P2_A, Actions.P2_B, Actions.P2_C, Actions.P2_D],
        8: [],
        9: [Actions.P2_D, Actions.P2_D]
    }[action]

# The Street Fighter specific interface for training an agent against the game
class Environment(gym.Env):

    # env_id - the unique identifier of the emulator environment, used to create fifo pipes
    # difficulty - the difficult to be used in story mode gameplay
    # frame_ratio, frames_per_step - see Emulator class
    # render, throttle, debug - see Console class
    def __init__(self, env_id, roms_path, difficulty=3, frame_ratio=3, frames_per_step=3, 
                 render=True, throttle=False, frame_skip=0, sound=False,
                 debug=False, binary_path=None, screen_id=0, project_dir=""):
        self.project_dir = project_dir
        self.difficulty = difficulty
        self.frame_ratio = frame_ratio
        self.frames_per_step = frames_per_step
        self.throttle = throttle
        self.emu = Emulator(env_id, roms_path, "kof97", 
                            setup_memory_addresses(), 
                            debug=False, frame_ratio=frame_ratio, 
                            render=render, frame_skip=frame_skip)
        self.started = False
        self.expected_health = {"P1": 0, "P2": 0}
        self.expected_wins = {"P1": 0, "P2": 0}
        self.stage = 1
        self.prev_action = []
        self.prev_playing = 0
        self.status = []
        self.loaded = False
        self.render_mode = "rgb_array"

    # Runs a set of action steps over a series of time steps
    # Used for transitioning the emulator through non-learnable gameplay, aka. title screens, character selects
    def run_steps(self, steps):
        for step in steps:
            for i in range(step["wait"]):
                self.emu.step([])
            self.emu.step([action.value for action in step["actions"]])

    def start(self, status=None):
        # self.run_steps(start_game_2p(self.frame_ratio))
        # self.emu.console.writeln(f'manager:machine():load("/home/jovyan/myc/embodied_project/kof_ai_v3/status/mai_2p_started_play")')
        if status is not None:
            self.emu.console.writeln(f'manager:machine():load("{status}")')
        else:
            # self.run_steps(start_game_2p_mai(self.frame_ratio))
            # path = sorted(glob.glob("/ssd/myc/Embodied_project/AIbotForKof97/status/*"))[-1]
            # self.stage = int(path.split('_')[-1])
            # self.emu.console.writeln(f'manager:machine():load("{path}")')
            self.emu.console.writeln(f'manager:machine():load("{self.project_dir}/status/mai_2p_started")')
        data = self.wait_for_fight_start()
        self.started = True
        self.done = False
        self.status = []
        return data['frame']
    
    def reset_display(self, status=None):
        if not self.loaded:
            self.emu.console.writeln(f'manager:machine():load("{status}")')
            self.loaded = True
        aa = [[index_to_attack_action_p2(0)[0]], []][random.randint(0, 1)]
        data = self.gather_frames(aa)
        self.last_info = data
        if data['playing'] == 32:
            self.expected_health = {"P1": data["healthP1"], "P2": data["healthP2"]}
            data = self.gather_frames([])
            data["action"] = 8
            data["reward"] = 0
            data["toward"] = 0
            data["damage_reward"] = 0
            data["distance_reward"] = 0
            data["time_reward"] = 0
            data["pow_reward"] = 0
            self.last_info = data
            self.started = True
            self.done = False
            self.status = []
            self.loaded = False
        return data['frame'], data['playing']

    
    def seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def next_stage(self):
        self.run_steps(next_stage(self.frame_ratio))
        self.stage = self.stage + 1
        path = f"{self.project_dir}/status/wait_for_fight_start_stage_%02d"%(self.stage)
        self.emu.console.writeln(f'manager:machine():save("{path}")')
        data = self.wait_for_fight_start()
        self.started = True
        self.done = False
        return data['frame']

    # Observes the game and waits for the fight to start
    def wait_for_fight_start(self):
        data = self.emu.step([])
        i = 0
        while data['playing'] != 32:
            aa = [[index_to_attack_action_p1(0)[0].value], []][random.randint(0, 1)]
            data = self.emu.step(aa)
        self.expected_health = {"P1": data["healthP1"], "P2": data["healthP2"]}
        data = self.gather_frames([])
        data["action"] = 8
        data["reward"] = 0
        data["toward"] = 0
        data["damage_reward"] = 0
        data["distance_reward"] = 0
        data["time_reward"] = 0
        data["pow_reward"] = 0
        self.last_info = data
        return data
    
    def process_toward(self, data):
        last_toward = self.last_info["toward"]
        toward = last_toward
        if int(last_toward) == 0 :
            if data["1P_x"] <= data["2P_x"]:
                toward = 0
            else:
                toward = 1
        else:
            if data["1P_x"] >= data["2P_x"]:
                toward = 1
            else:
                toward = 0
        data["toward"] = toward

    def reset(self, status=None, **kwargs):
        return self.start(status=status)

    # Steps the emulator along until the screen goes black at the very end of a game
    def wait_for_continue(self):
        data = self.emu.step([])
        while data['playing'] != 32:
            ma = random.randint(0, 1)
            aa = random.randint(0, 1)
            data = self.emu.step([index_to_move_action_p1(ma)[0].value, index_to_attack_action_p1(aa)[0].value])
        self.done = False

    # Checks whether the round or game has finished
    def check_done(self, data):
        if data["playing"] == 32:
            data['done'] = False
            data['win'] = 'UNKNOWN'
        else:
            data['done'] = True
            if data['healthP1'] < data['healthP2']:
                data['win'] = "FALSE"
            else:
                data['win'] = "TRUE"
            self.status.append(data['win'])

    # Collects the specified amount of frames the agent requires before choosing an action
    def gather_frames(self, actions):
        data = self.sub_step(actions)
        frames = [data["frame"]]
        for i in range(self.frames_per_step - 1):
            data = add_rewards(data, self.sub_step(actions))
            frames.append(data["frame"])
        data["frame"] = frames[0] if self.frames_per_step == 1 else frames
        return data

    # Steps the emulator along by one time step and feeds in any actions that require pressing
    # Takes the data returned from the step and updates book keeping variables
    def sub_step(self, actions):
        data = self.emu.step([action.value for action in actions])

        p1_diff = (self.expected_health["P1"] - data["healthP1"])
        p2_diff = (self.expected_health["P2"] - data["healthP2"])
        self.expected_health = {"P1": data["healthP1"], "P2": data["healthP2"]}

        rewards = {
            "P1": (p2_diff-p1_diff),
            "P2": (p1_diff-p2_diff)
        }

        data["rewards"] = rewards
        return data

    # Steps the emulator along by the requested amount of frames required for the agent to provide actions
    def step(self, action, mode='computer'):
        if self.started:
            if not self.done:
                if mode == 'human':
                    data = self.gather_frames(action)
                    self.postprocess(data)
                    self.check_done(data)
                    self.done = data['done']
                    # self.print_log(data, [move_action, attack_action])
                    self.process_toward(data)
                    self.last_info = data
                    return data["frame"], data["rewards"], self.done, data
                elif mode == "computer":
                    move_action, attack_action = action
                    actions = []
                    actions += index_to_move_action_p1(move_action)
                    actions += index_to_attack_action_p1(attack_action)
                    data = self.gather_frames(actions)
                    self.postprocess(data)
                    self.check_done(data)
                    self.done = data['done']
                    # self.print_log(data, [move_action, attack_action])
                    self.process_toward(data)
                    self.last_info = data
                    return data["frame"], data["rewards"], self.done, data
                elif mode == "computer_2p":
                    move_action_1, move_action_2, attack_action_1, attack_action_2 = action
                    actions = []
                    actions += index_to_move_action_p1(move_action_1)
                    actions += index_to_move_action_p2(move_action_2)
                    actions += index_to_attack_action_p1(attack_action_1)
                    actions += index_to_attack_action_p2(attack_action_2)
                    data = self.gather_frames(actions)
                    self.postprocess(data)
                    self.check_done(data)
                    self.done = data['done']
                    # self.print_log(data, [move_action, attack_action])
                    self.process_toward(data)
                    self.last_info = data
                    return data["frame"], data["rewards"], self.done, data
                elif mode == "interaction":
                    data = self.gather_frames(action)
                    self.postprocess(data)
                    self.check_done(data)
                    self.done = data['done']
                    # self.print_log(data, [move_action, attack_action])
                    self.process_toward(data)
                    self.last_info = data
                    return data["frame"], data["rewards"], self.done, data
                
            else:
                data = self.gather_frames(action)
                self.postprocess(data)
                self.check_done(data)
                self.done = data['done']
                # self.print_log(data, [move_action, attack_action])
                self.process_toward(data)
                self.last_info = data
                return data["frame"], data["rewards"], self.done, data

        else:
            raise EnvironmentError("Start must be called before stepping")
        
    def split_hex(self, data, num_byte=4):
        hex_data = f"{data:08x}"
        return [hex_data[2*i:2*(i+1)] for i in range(num_byte)]
    
    def split_bit(self, data):
        return list(f"{data:08b}")

    def postprocess(self, data):
        P1_attackstatus = self.split_hex(data['1P_AttackStatus'], 4)
        p1_status = self.split_bit(data['1P_Status'])
        # p1_status_extra = self.split_bit(data['1P_StatusExtra'])
        # p1_power_status = self.split_bit(data['1P_PowerStatus'])

        P2_attackstatus = self.split_hex(data['2P_AttackStatus'], 4)
        p2_status = self.split_bit(data['2P_Status'])
        # p2_status_extra = self.split_bit(data['2P_StatusExtra'])
        # p2_power_status = self.split_bit(data['2P_PowerStatus'])

        data['p1_pownum']  = data['1P_PowerValue']
        data['p1_acting'] = (p1_status[-1] == '1') or (p1_status[-2] == '1') or (p1_status[-3] == '1')
        data['p1_airing'] = p1_status[-2] == '1'
        data['p1_powing'] = p1_status[-5] == '1'
        # 0 未接触 1 攻击成功 -1 攻击失败
        data['p1_attack'] = 0 if P2_attackstatus[2] == '00' else (1 if P2_attackstatus[2] != '0a' else -1)
        # 0 未接触 1 防御成功 -1 防御失败
        data['p1_defense'] = 0 if P1_attackstatus[2] == '00' else (-1 if P1_attackstatus[2] != '0a' else 1)

        data['touch'] = P1_attackstatus[2] != '00' or P2_attackstatus[2] != '00'

        data['p2_pownum']  = data['2P_PowerValue']
        data['p2_acting'] = (p2_status[-1] == '1') or (p2_status[-2] == '1') or (p2_status[-3] == '1')
        data['p2_airing'] = p2_status[-2] == '1'
        data['p2_powing'] = p2_status[-5] == '1'
        # 0 未接触 1 攻击成功 -1 攻击失败
        data['p2_attack'] = 0 if P1_attackstatus[2] == '00' else (1 if P1_attackstatus[2] != '0a' else -1)
        # 0 未接触 1 防御成功 -1 防御失败
        data['p2_defense'] = 0 if P2_attackstatus[2] == '00' else (-1 if P2_attackstatus[2] != '0a' else 1)

        # print("[1P] power number: %d, acting: %d, airing: %d, powing: %d  "%
        #       (data['p1_pownum'], data['p1_acting'], data['p1_airing'], data['p1_powing'])+
        #       "[2P] power number: %d, acting: %d, airing: %d, powing: %d"% 
        #        (data['p2_pownum'], data['p2_acting'], data['p2_airing'], data['p2_powing'])
        #        )
        # print(p1_status)
        # if data['1P_AttackStatus'] != 0 or data['2P_AttackStatus'] != 0:
        #     print("######################################")
        #     print("1P 动作中：%d, 攻击成功：%d, 防御成功%d"%(data['p1_acting'], 
        #                                             data['p1_attack'], 
        #                                             data['p1_defense']))
            
        #     print("2P 动作中：%d, 攻击成功：%d, 防御成功%d"%(data['p2_acting'], 
        #                                             data['p2_attack'], 
        #                                             data['p2_defense']))
        #     print("######################################")

    def print_log(self, data, action):
        print("stage: %02d, done: %1d, win: %1d/%1d, time: %03d, playing: %03d, hp1: %.3d, hp2: %03d"%(
            self.stage, data['done'], self.status.count('TRUE'), 
            self.status.count('FALSE'), 
            data['time'], data['playing'], 
            data['healthP1'], data['healthP2']))
                
    def render(self, mode):
        if hasattr(self, "last_info"):
            return self.last_info["frame"]
        else:
            return np.zeros([224, 320, 3])
        
    def get_keys_to_action_p2(self):
        return {
            'w': Actions.P2_UP,
            'd': Actions.P2_RIGHT,
            's': Actions.P2_DOWN,
            'a': Actions.P2_LEFT,
            'j': Actions.P2_A,
            'k': Actions.P2_B,
            'l': Actions.P2_C,
            'u': Actions.P2_D,
        }
    def get_keys_to_action_p1(self):
        return {
            'w': Actions.P1_UP,
            'd': Actions.P1_RIGHT,
            's': Actions.P1_DOWN,
            'a': Actions.P1_LEFT,
            'j': Actions.P1_A,
            'k': Actions.P1_B,
            'l': Actions.P1_C,
            'u': Actions.P1_D,
        }

    # Safely closes emulator
    def close(self):
        self.emu.close()
