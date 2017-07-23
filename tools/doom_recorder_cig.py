#
# doom_recorder_cig.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import datetime
import glob
import h5py
from vizdoom import *
import argparse


class DoomRecorder:
    def __init__(self, config, wad, skiprate, h5_path):
        self.game = DoomGame()
        self.game.set_doom_game_path(wad)
        self.game.load_config(config)
        self.game.set_sound_enabled(True)
        #self.game.add_game_args("+freelook 1")
        self.game.add_game_args("-host 1 -deathmatch +timelimit 5.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                           "+viz_respawn_delay 0 +viz_nocheat 1")
        self.game.add_game_args("+name DoomNet +colorset 0")
        self.skiprate = skiprate
        self.h5_path = h5_path

    def play(self):
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_window_visible(True)
        self.game.set_mode(Mode.SPECTATOR)
        self.game.init()

        for episode in range(1):
            timestamp = '{:%Y-%m-%d %H-%M-%S}'.format(datetime.datetime.now())
            self.game.new_episode()
            self.game.send_game_command("removebots")
            for i in range(10):
                self.game.send_game_command("addbot")

            screens = []
            actions = []
            rewards = []
            variables = []
            depths = []
            labels = []
            automaps = []

            for i in range(10000):
                state = self.game.get_state()

                self.game.advance_action(self.skiprate)
                action = self.game.get_last_action()
                reward = self.game.get_last_reward()

                if not all(a == 0 for a in action):  # save only if an action is taken
                    screens.append(state.screen_buffer)
                    actions.append(action)
                    rewards.append(reward)
                    if state.game_variables is not None:
                        variables.append(state.game_variables)
                    if state.depth_buffer is not None:
                        depths.append(state.depth_buffer)
                    if state.labels_buffer is not None:
                        labels.append(state.labels_buffer)
                    if state.automap_buffer is not None:
                        automaps.append(state.automap_buffer)

                print("State #" + str(state.number))
                print("Game variables: ", state.game_variables)
                print("Action:", action)
                print("Reward:", reward)
                print("=====================")
                if reward != 0:
                    print(reward)

                if self.game.is_player_dead():
                    state = self.game.get_state()
                    self.game.respawn_player()

                if self.game.is_episode_finished():
                    break

            print("Episode finished!")
            print("Total reward:", self.game.get_total_reward())
            print("************************")

            print("writing episode...")
            with h5py.File(self.h5_path + timestamp + '.h5', 'w') as file:
                file.create_dataset("screens", data=screens, compression='gzip')
                file.create_dataset("actions", data=actions, compression='gzip')
                file.create_dataset("rewards", data=rewards, compression='gzip')
                if variables:
                    file.create_dataset("variables", data=variables, compression='gzip')
                if depths:
                    file.create_dataset("depths", data=depths, compression='gzip')
                if labels:
                    file.create_dataset("labels", data=labels, compression='gzip')
                if automaps:
                    file.create_dataset("automaps", data=automaps, compression='gzip')

        self.game.close()

    def replay(self):
        # New render settings for replay
        self.game.set_screen_resolution(ScreenResolution.RES_800X600)
        self.game.set_window_visible(True)
        self.game.set_render_hud(True)

        self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.init()

        print("\nREPLAY OF EPISODE")
        print("************************\n")

        for episode_file in glob.glob(self.h5_path + '*.h5'):
            print("reading episode...")
            with h5py.File(episode_file, 'r') as file:
                screens = file['screens'][:]
                actions = file['actions'][:]
                rewards = file['rewards'][:]

            # Replays episodes stored in given file. Sending game command will interrupt playback.
                self.game.new_episode()
            for a in actions:
                s = self.game.get_state()
                # Use advance_action instead of make_action.
                r = self.game.make_action(a.tolist(), self.skiprate)

                # game.get_last_action is not supported and don't work for replay at the moment.

                print("State #" + str(s.number))
                print("Game variables:", s.game_variables[0])
                print("Reward:", r, a)
                print("=====================")

            print("Episode finished.")
            print("total reward:", self.game.get_total_reward())
            print("************************")

        self.game.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Doom Recorder')
    # parser.add_argument('--vizdoom_config', default='../environments/health_gathering.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_config', default='../environments/cig2.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_path', default=os.path.expanduser('~') + '/tools/ViZDoom/bin/vizdoom',
                        help='path to vizdoom')
    parser.add_argument('--wad_path', default=os.path.expanduser('~') + '/tools/ViZDoom/scenarios/Doom2.wad',
                        help='wad file path')
    parser.add_argument('--h5_path', default=os.path.expanduser('~') + '/test/datasets/vizdoom/cig_map02/',
                        help='hd5 files path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')

    args = parser.parse_args()

    #a = list(map(int, bin(10)[2:]))

    recorder = DoomRecorder(args.vizdoom_config, args.wad_path, args.skiprate, args.h5_path)

    recorder.play()
    #recorder.replay()
