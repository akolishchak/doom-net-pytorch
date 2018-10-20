#
# bt_doom.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
#
import os
import time
from bt import BTNode, BTSequence, BTFallback, BTInverter
from doom_object import DoomObject
from bt_doom_actions import Open, Goto, Pick, Attack
from bt_doom_context import Context
from doom_instance_bt import DoomInstanceBt
import vizdoom

# goal_condition, conditions, action, parameters
# conditions


class ConditionNode(BTNode):
    def __init__(self, name, condition):
        super().__init__(name)
        self.condition = condition

    def run(self, context):
        if self.condition(context):
            return self.Result.Success
        else:
            return self.Result.Failure


conditions = [
    ConditionNode(
        'near_exit',
        lambda context: context.game_state.objects[context.game_state.objects == DoomObject.Type.EXIT].size > 0
    ),
    ConditionNode(
        'near_enemy',
        lambda context: context.game_state.objects[context.game_state.objects == DoomObject.Type.ENEMY].size > 0
    ),
    ConditionNode(
        'near_door',
        lambda context: context.game_state.objects[context.game_state.objects.shape[0]//2] == DoomObject.Type.DOOR
    ),
    ConditionNode(
        'near_weapon',
        lambda context: context.game_state.objects[context.game_state.objects == DoomObject.Type.WEAPON].size > 0
    ),
    ConditionNode(
        'near_health',
        lambda context: context.game_state.objects[context.game_state.objects == DoomObject.Type.HEALTH].size > 0 and
                        context.game_state.variables[0] < 100
    ),
    ConditionNode(
        'near_ammo',
        lambda context: context.game_state.objects[context.game_state.objects == DoomObject.Type.AMMO].size > 0
    ),
    ConditionNode(
        'finished',
        lambda context: context.is_finished
    ),
    ConditionNode(
        'enough_health',
        lambda context: context.game_state.variables[0] > 30 or
                        context.game_state.objects[context.game_state.objects == DoomObject.Type.HEALTH].size == 0
    ),
    ConditionNode(
        'enough_ammo',
        lambda context: context.game_state.variables[2] > 1 or
                        context.game_state.objects[context.game_state.objects == DoomObject.Type.AMMO].size == 0
    ),
    ConditionNode(
        'health_exist',
        lambda context: not context.level_map.get_health()
    ),
    ConditionNode(
        'ammo_exist',
        lambda context: not context.level_map.get_ammo()
    ),
]
conditions = {condition.name: condition for condition in conditions}

conditions['not_near_enemy'] = BTInverter(conditions['near_enemy'])
conditions['not_near_health'] = BTInverter(conditions['near_health'])
conditions['not_near_ammo'] = BTInverter(conditions['near_ammo'])
conditions['not_near_weapon'] = BTInverter(conditions['near_weapon'])
conditions['not_near_door'] = BTInverter(conditions['near_door'])


actions = [
    Open('open_exit', DoomObject.Type.EXIT),
    Open('open_door', DoomObject.Type.DOOR),
    Goto('goto_exit', DoomObject.Type.EXIT),
    Goto('goto_health', DoomObject.Type.HEALTH),
    Goto('goto_ammo', DoomObject.Type.AMMO),
    Pick('pick_health', DoomObject.Type.HEALTH),
    Pick('pick_ammo', DoomObject.Type.AMMO),
    #Pick('pick_key', DoomObject.Type.KEY),
    Pick('pick_weapon', DoomObject.Type.WEAPON),
    Attack('attack'),
]
actions = {action.name: action for action in actions}


class ActionDef:
    def __init__(self, pre_conditions, action):
        self.pre_conditions = pre_conditions
        self.action = action


goal_defs = {
    'finished': ActionDef(
                    ['near_exit'],
                    'open_exit'
                ),
    'near_exit': ActionDef(
                    ['not_near_enemy', 'not_near_health', 'not_near_ammo', 'not_near_weapon', 'not_near_door'],
                    'goto_exit'
                ),
    'not_near_enemy': ActionDef(
                    ['near_enemy', 'enough_health', 'enough_ammo'],
                    'attack'
                ),
    'enough_health': ActionDef(
                    ['near_health'],
                    'pick_health'
                ),
    #'near_health': ActionDef(
    #                ['health_exist'],
    #                'goto_health'
    #            ),
    'enough_ammo': ActionDef(
                    ['near_ammo'],
                    'pick_ammo'
                ),
    #'near_ammo': ActionDef(
    #                ['ammo_exist'],
    #                'goto_ammo'
    #            ),
    'not_near_health': ActionDef(
                    ['near_health'],
                    'pick_health'
                ),
    'not_near_ammo': ActionDef(
                    ['near_ammo'],
                    'pick_ammo'
                ),
    'not_near_weapon': ActionDef(
                    ['near_weapon'],
                    'pick_weapon'
                ),
    'not_near_door': ActionDef(
                    ['near_door'],
                    'open_door'
                ),
}


class BTBulder:
    def __init__(self, conditions, actions, goal_defs):
        self.conditions = conditions
        self.actions = actions
        self.goal_defs = goal_defs

    def expand(self, goal_name):
        goal = self.conditions[goal_name]

        try:
            definition = self.goal_defs[goal_name]
        except KeyError:
            return goal

        condition_nodes = []
        for condition in definition.pre_conditions:
            node = self.expand(condition)
            condition_nodes.append(node)

        action = self.actions[definition.action]
        action_node = BTSequence(condition_nodes + [action])
        node = BTFallback([goal, action_node])
        return node


def test():

    builder = BTBulder(conditions, actions, goal_defs)
    bt = builder.expand('finished')
    bt.draw('bt_doom_root.png')

    vizdoom_path = os.path.dirname(vizdoom.__file__)
    config = "../environments/oblige/oblige-map.cfg"

    game_levels = DoomInstanceBt.get_game_levels(config)
    print('Game levels: ', len(game_levels))
    for i, [wad_file, map_id] in enumerate(game_levels):
        print('Playing ''{}'', map{:02d}'.format(wad_file, map_id+1))
        game = DoomInstanceBt(config,
                              vizdoom_path + "/freedoom2.wad",
                              skiprate=4,
                              visible=True,
                              actions=[],
                              id=0,
                              config_wad=wad_file,
                              map_id=map_id
                              )

        while True:
            context = Context(game)
            Result = bt.run(context)
            time.sleep(0.08)
            if Result == BTNode.Result.Success:
                break

        print('Game completed.')
        game.release()

test()





