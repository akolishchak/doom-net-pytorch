#
# bt_doom_context.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
#


class Context:
    def __init__(self, game):
        self.game = game
        self.game_state = game.get_state_normalized()
        self.is_finished = game.is_finished()
        self.level_map = game.level_map
        self.pose = game.get_pose()
        self.object_info = game.get_object_info(self.game_state)

    def game_step(self, action):
        action[2] /= self.game.skiprate
        return self.game.step_normalized(action)

