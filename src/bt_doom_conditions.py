#
# bt_doom_conditions.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
#
from bt import BTNode


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

