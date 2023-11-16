from enum import Enum
from MAMEToolkit.emulator import Action


# An enumerable class used to specify which actions can be used to interact with a game
# Specifies the Lua engine port and field names required for performing an action
class Actions(Enum):
    # Starting
    SERVICE = Action(':TEST', 'Service Mode')

    COIN_P1 = Action(':AUDIO/COIN', 'Coin 1')
    COIN_P2 = Action(':AUDIO/COIN', 'Coin 2')

    P1_START = Action(':edge:joy:START', '1 Player Start')
    P2_START = Action(':edge:joy:START', '2 Players Start')

    # Movement
    P1_UP = Action(':edge:joy:JOY1', 'P1 Up')
    P1_DOWN = Action(':edge:joy:JOY1', 'P1 Down')
    P1_LEFT = Action(':edge:joy:JOY1', 'P1 Left')
    P1_RIGHT = Action(':edge:joy:JOY1', 'P1 Right')

    P2_UP = Action(':edge:joy:JOY2', 'P2 Up')
    P2_DOWN = Action(':edge:joy:JOY2', 'P2 Down')
    P2_LEFT = Action(':edge:joy:JOY2', 'P2 Left')
    P2_RIGHT = Action(':edge:joy:JOY2', 'P2 Right')

    # Fighting
    P1_A = Action(':edge:joy:JOY1', 'P1 Button 1')
    P1_B = Action(':edge:joy:JOY1', 'P1 Button 2')
    P1_C = Action(':edge:joy:JOY1', 'P1 Button 3')
    P1_D = Action(':edge:joy:JOY1', 'P1 Button 4')

    P2_A = Action(':edge:joy:JOY2', 'P2 Button 1')
    P2_B = Action(':edge:joy:JOY2', 'P2 Button 2')
    P2_C = Action(':edge:joy:JOY2', 'P2 Button 3')
    P2_D = Action(':edge:joy:JOY2', 'P2 Button 4')
