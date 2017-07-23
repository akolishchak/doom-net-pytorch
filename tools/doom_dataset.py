#
# doom_dataset.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import os
import numpy as np
import glob
import h5py
import argparse


class DoomDataset:
    def __init__(self, h5_path, data_keys=['screens', 'depths', 'labels', 'variables', 'automaps', 'rewards']):
        self.h5_path = h5_path
        self.length = 0
        self.data_keys = data_keys
        self.episodes = []
        self.file_names = []
        actions = []
        for episode_file in glob.glob(h5_path + '*.h5'):
            print("reading episode", episode_file)
            file = h5py.File(episode_file, 'r')
            episode_length = file['screens'].len()
            episode = {}
            for key in self.data_keys:
                episode[key] = file[key] if key in file else None
            episode['length'] = episode_length
            self.episodes.append(episode)
            self.file_names.append(episode_file)
            actions.append(file['actions'][:])
            self.length += episode_length
            #break
        # prepare actions
        actions = np.concatenate(actions).astype(np.uint8)
        """
        """
        #
        # cig.cfg only corrections
        #
        # remove other actions from shots (idx = 2)
        actions[actions[:, 2] == 1] = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        #
        actions[:, 8] = 0
        """
        # replace TURN_LEFT_RIGHT_DELTA by TURN_RIGHT
        actions[actions[:, 7] != 0, 1] = 1
        actions[actions[:, 7] != 0, 7] = 0
        # replace MOVE_FORWARD+MOVE_BACKWARD by MOVE_BACKWARD
        actions[((actions[:, 0] == 0) &
                (actions[:, 1] == 0) &
                (actions[:, 2] == 0) &
                (actions[:, 3] == 0) &
                (actions[:, 4] == 0) &
                (actions[:, 5] == 1) &
                (actions[:, 6] == 1) &
                (actions[:, 7] == 0) &
                (actions[:, 8] == 0))] = [0, 0, 0, 0, 0, 0, 1, 0, 0]
        # remove MOVE_FORWARD+MOVE_BACKWARD everywhere
        idx = (actions[:, 5] == 1) & (actions[:, 6] == 1)
        actions[idx, 5] = 0
        actions[idx, 6] = 0
        # replace TURN_LEFT+TURN_RIGHT by MOVE_BACKWARD
        actions[((actions[:, 0] == 1) &
                (actions[:, 1] == 1) &
                (actions[:, 2] == 0) &
                (actions[:, 3] == 0) &
                (actions[:, 4] == 0) &
                (actions[:, 5] == 0) &
                (actions[:, 6] == 0) &
                (actions[:, 7] == 0) &
                (actions[:, 8] == 0))] = [0, 0, 0, 0, 0, 0, 1, 0, 0]
        # remove TURN_LEFT+TURN_RIGHT everywhere
        idx = (actions[:, 0] == 1) & (actions[:, 1] == 1)
        actions[idx, 0] = 0
        actions[idx, 1] = 0
        """
        #
        #
        #
        flat_actions = actions.view(np.dtype((np.void, actions.dtype.itemsize * actions.shape[1]))).squeeze()
        unique_flat_actions = np.unique(flat_actions)
        self.action_sets = unique_flat_actions.view(actions.dtype).reshape(-1, actions.shape[1])
        self.action_labels = np.zeros(actions.shape[0], dtype=np.uint8)
        for i in range(len(self.action_sets)):
            self.action_labels[flat_actions == unique_flat_actions[i]] = i
        """
        """
        #self.action_sets = np.append(self.action_sets, [[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]], axis=0)
        #self.action_sets[7] = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        #self.action_sets[4] = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        #self.action_sets[:, 7] = 1

    def create_dataset(self, file, key):
        if self.episodes[0][key] is not None:
            shape = list(self.episodes[0][key].shape)
            shape[0] = self.length
            return file.create_dataset(key, shape=shape, dtype=self.episodes[0][key].dtype, compression='gzip')
        else:
            return None

    def create_flat(self, file_name):
        with h5py.File(self.h5_path + file_name, 'w') as file:
            print('create dataset...')
            file.create_dataset('action_labels', data=self.action_labels, compression='gzip')
            file.create_dataset('action_sets', data=self.action_sets, compression='gzip')

            data = {}
            for key in self.data_keys:
                episode = self.episodes[0][key]
                if episode is not None:
                    shape = list(episode.shape)
                    shape[0] = self.length
                    data[key] = file.create_dataset(key, shape=shape, dtype=episode.dtype) #, compression='gzip')
                else:
                    data[key] = None

            episode_offsets = []
            offset = 0
            for episode, file_name in zip(self.episodes, self.file_names):
                print(file_name)
                episode_length = episode['length']
                print('episode offset: {}, length: {}'.format(offset, episode_length))
                for key in self.data_keys:
                    print('write {}...'.format(key))
                    if data[key] is not None:
                        data[key][offset:offset+episode_length] = episode[key][:]
                # get episode offsets
                episode_start = offset
                for episode_end in np.where(episode['variables'][:, 0] < 0)[0]:
                    episode_offsets.append([episode_start, offset+episode_end-1])
                    print(episode_offsets[-1], offset, episode_start, episode_end)
                    episode_start = offset+episode_end+1
                offset += episode_length
                if episode_start != offset:
                    episode_offsets.append([episode_start, offset-1])
                    print(episode_offsets[-1], offset, episode_start)
                else:
                    episode_start += 1

            file.create_dataset('episodes', data=episode_offsets, compression='gzip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Doom Dataset')
    parser.add_argument('--h5_path', default=os.path.expanduser('~') + '/test/datasets/vizdoom/bobby_cig_map01/',
                        help='hd5 files path')

    args = parser.parse_args()

    dataset = DoomDataset(args.h5_path, ['screens', 'variables', 'rewards'])
    dataset.create_flat('flat.h5')

