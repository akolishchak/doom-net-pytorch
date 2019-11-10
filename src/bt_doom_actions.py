#
# bt_doom_actions.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
#
import random
from bt import BTNode
from doom_object import DoomObject

#
# actions
#
class Open(BTNode):
    def __init__(self, name, object_type):
        super().__init__(name)
        self.object_type = object_type

    def run(self, context):
        for type, distance, angle in context.object_info:
            if type == self.object_type:
                if distance <= 2:
                    state, _, _ = context.game_step([100, 0, angle, 0, 1])
                    i = 0
                    while state.objects[state.objects == DoomObject.Type.DOOR].size > 0 and i < 10:
                        state, _, _ = context.game_step([100, 0, 0, 0, 0])
                        i += 1
                else:
                    context.game_step([100, 0, angle, 0, 0])
                return self.Result.Failure

        return self.Result.Success


class Goto(BTNode):
    def __init__(self, name, object_type):
        super().__init__(name)
        self.object_type = object_type

    def run(self, context):
        if self.object_type == DoomObject.Type.EXIT:
            heading = context.level_map.get_exit_heading(context.pose[DoomObject.Y], context.pose[DoomObject.X])
            headling_error = context.pose[DoomObject.HEADING] - heading
            # randomize heading
            headling_error += 2*(random.random() - 0.5)*0.5*headling_error
            context.game_step([100, 0, headling_error, 0, 0])
            return self.Result.Failure
        else:
            return self.Result.Success


class Pick(BTNode):
    def __init__(self, name, object_type):
        super().__init__(name)
        self.object_type = object_type

    def run(self, context):
        for type, distance, angle in context.object_info:
            if type == self.object_type:
                context.game_step([100, 0, angle, 0, 0])
                return self.Result.Failure

        return self.Result.Success


class Attack(BTNode):
    def __init__(self, name):
        super().__init__(name)

    def run(self, context):
        for type, distance, angle in context.object_info:
            if type == DoomObject.Type.ENEMY:
                # shoot
                context.game_step([0, 0, angle, 1, 0])
                # change position
                shift = 100 if random.random() > 0.5 else -100
                for i in range(1):
                    context.game_step([0, shift, 0, 0, 0])
                return self.Result.Failure

        return self.Result.Success
