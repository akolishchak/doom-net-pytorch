#
# doom_recorder.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import glob
import argparse
from vizdoom import *

def replay(config, wad, skiprate, path):
    game = DoomGame()
    game.set_doom_game_path(wad)
    game.load_config(config)

    game.set_screen_resolution(ScreenResolution.RES_800X600)
    game.set_window_visible(True)
    game.set_render_hud(True)

    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for episode_file in glob.glob(path + '*.lmp'):
        print('replay episode:', episode_file)
        game.replay_episode(episode_file)
        while not game.is_episode_finished():
            state = game.get_state()
            game.advance_action(skiprate)
            reward = game.get_last_reward()
            print('State #{}: reward = {}'.format(state.number, reward))

        print('total reward:', game.get_total_reward())

    game.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Doom Recorder')
    parser.add_argument('--vizdoom_config', default='../environments/health_gathering.cfg', help='vizdoom config path')
    parser.add_argument('--vizdoom_path', default=os.path.expanduser('~') + '/tools/ViZDoom/bin/vizdoom',
                        help='path to vizdoom')
    parser.add_argument('--wad_path', default=os.path.expanduser('~') + '/tools/ViZDoom/scenarios/Doom2.wad',
                        help='wad file path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')
    parser.add_argument('--path', default='../', help='.lmp files path')

    args = parser.parse_args()

    replay(args.vizdoom_config, args.wad_path, args.skiprate, args.path)
