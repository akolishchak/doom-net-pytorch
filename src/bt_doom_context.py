#
# bt_doom_context.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
#


class Context:
    def __init__(self, game):
        self.game = game
        self.level_map = game.level_map
        self.targets = []
        self.game_state = None
        self.is_finished = None
        self.pose = None
        self.object_info = None

    def update(self):
        self.game_state = self.game.get_state_normalized()
        self.is_finished = self.game.is_finished()
        self.pose = self.game.get_pose()
        self.object_info = self.game.get_object_info(self.game_state)

    def game_step(self, action):
        action[2] /= self.game.skiprate
        return self.game.step_normalized(action)
