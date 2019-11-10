#
# doom_instance_cig.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
from vizdoom import *
from doom_instance import DoomInstance
import numpy as np


class DoomInstanceCig():
    def __init__(self, config, wad, skiprate, visible=False, mode=Mode.PLAYER, actions=None, id=None, color=0, bot_num=10):
        self.bot_num = bot_num
        args = (
            "+name DoomNet +colorset ".format(color) +
            " -deathmatch"
            " +sv_forcerespawn 0"
            " +sv_noautoaim 1"
            " +sv_respawnprotect 1"
            " +sv_spawnfarthest 1"
            " +sv_nocrouch 1"
            " +sv_nocrouch 1"
            " +viz_respawn_delay 0"
            #" +viz_nocheat 1"
            #" +viz_debug 0"
            " +timelimit 10.0"
        )
        super().__init__(config, wad, skiprate, visible, mode, actions, id, args)

    def step_normalized(self, action):
        state, reward, finished = super().step_normalized(action)

        # comment this for basic and rocket configs
        if state.variables is not None:
            diff = state.variables - self.variables
            #if diff[1] < -100:
            #    diff[1] = 0
            diff = np.multiply(diff, [100 * 0.5 * (0.2 if diff[0] > 0 else 0.02), 100 * 0.5 * 0.01, 100 * 1 * 1])
            if diff[2] > 0:
                print('HIT!!!', self.id)
            #if dead:
            #    diff[2] = -100
            #    print('DEAD', self.id)
            # penalize shots with zero ammo
            if self.variables[0] == 0 and self.actions[action][2] == 1:
                diff[0] -= 10
            reward += diff.sum() - 3

            self.variables = state.variables.copy()
            state.variables[2] = 0

        return state, reward, finished

    def new_episode(self):
        super().new_episode()
        self.game.send_game_command("removebots")
        if self.id is not None:
            for i in range(self.bot_num):
                self.game.send_game_command("addbot")
