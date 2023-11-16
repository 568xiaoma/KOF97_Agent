from actions import Actions

def start_game(frame_ratio):
    return [
        {"wait": int(300/frame_ratio), "actions": [Actions.COIN_P1]},
        {"wait": int(30/frame_ratio), "actions": [Actions.P1_START]},
        {"wait": int(120/frame_ratio), "actions": [Actions.P1_A]}, # 确认进入
        {"wait": int(180/frame_ratio), "actions": [Actions.P1_A]}, # 选择模式
        {"wait": int(120/frame_ratio), "actions": [Actions.P1_A]}, # 选人1
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A]}, # 选人2
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN]},
        {"wait": int(30/frame_ratio), "actions": [Actions.P1_A]}, # 选人2
        {"wait": int(200/frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_A]}
    ]
def start_game_2p(frame_ratio):
    return [
        {"wait": int(300/frame_ratio), "actions": [Actions.COIN_P1, Actions.COIN_P2]},
        {"wait": int(30/frame_ratio), "actions": [Actions.P1_START, Actions.P2_START]},
        {"wait": int(120/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 确认进入
        {"wait": int(180/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选择模式
        {"wait": int(120/frame_ratio), "actions": [Actions.P2_RIGHT, Actions.P2_UP]}, # 选人1
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选人1
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN, Actions.P2_DOWN]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选人2
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN, Actions.P2_DOWN]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN, Actions.P2_DOWN]},
        {"wait": int(30/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选人2
        {"wait": int(200/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}
    ]
def start_game_2p_mai(frame_ratio):
    return [
        {"wait": int(300/frame_ratio), "actions": [Actions.COIN_P1, Actions.COIN_P2]},
        {"wait": int(30/frame_ratio), "actions": [Actions.P1_START, Actions.P2_START]},
        {"wait": int(120/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 确认进入
        {"wait": int(180/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选择模式
        {"wait": int(120/frame_ratio), "actions": [Actions.P2_RIGHT, Actions.P2_UP]}, # 选人1
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选人1
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN, Actions.P2_DOWN]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选人2
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN, Actions.P2_DOWN]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN, Actions.P2_DOWN]},
        {"wait": int(30/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选人2
        {"wait": int(200/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}
    ]
def start_game_2p_iori(frame_ratio):
    return [
        {"wait": int(300/frame_ratio), "actions": [Actions.COIN_P1, Actions.COIN_P2]},
        {"wait": int(30/frame_ratio), "actions": [Actions.P1_START, Actions.P2_START]},
        {"wait": int(120/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 确认进入
        {"wait": int(180/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选择模式
        {"wait": int(120/frame_ratio), "actions": [Actions.P2_RIGHT, Actions.P2_UP]}, # 选人1
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选人1
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_RIGHT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN, Actions.P2_DOWN]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选人2
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN, Actions.P2_DOWN]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_LEFT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_DOWN, Actions.P2_DOWN]},
        {"wait": int(30/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}, # 选人2
        {"wait": int(200/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_RIGHT, Actions.P2_LEFT]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]},
        {"wait": int(20/frame_ratio), "actions": [Actions.P1_A, Actions.P2_A]}
    ]
    
def next_stage(frame_ratio):
    return [
        {"wait": int(300/frame_ratio), "actions": [Actions.COIN_P1]},
        {"wait": int(1100/frame_ratio), "actions": [Actions.P1_A]},
        {"wait": int(10/frame_ratio), "actions": [Actions.P1_A]},
    ]