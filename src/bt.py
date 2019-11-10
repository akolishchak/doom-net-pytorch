#
# bt.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
#
#import pygraphviz as pgv


class BTNode:
    class Result:
        Success = 1
        Failure = 2

    def __init__(self, name, nodes=[]):
        self.name = name
        self.nodes = nodes

    def run(self, context):
        assert not hasattr(super(), 'run')

    def draw(self, file_name):
        bt = self
        graph = pgv.AGraph()

        def id_gen():
            counter = 0
            while True:
                counter += 1
                yield '#' + str(counter) + '. '

        name_id_gen = id_gen()
        nodes = [[bt, next(name_id_gen) + bt.name]]

        while nodes:
            node, node_name = nodes.pop(0)
            for child in node.nodes:
                child_name = next(name_id_gen) + child.name
                graph.add_edge(node_name, child_name)
                nodes.append([child, child_name])

        graph.layout(prog='dot')
        graph.draw(file_name)


class BTSequence(BTNode):
    def __init__(self, nodes):
        super().__init__('-->', nodes)

    def run(self, context):
        for node in self.nodes:
            result = node.run(context)
            if result != self.Result.Success:
                return result

        return self.Result.Success

    def print(self):
        super().print()
        return self.name


class BTFallback(BTNode):
    def __init__(self, nodes):
        super().__init__(' ? ', nodes)

    def run(self, context):
        result = self.Result.Failure
        for node in self.nodes:
            result = node.run(context)
            if result == self.Result.Success:
                return result

        return result


class BTInverter(BTNode):
    def __init__(self, node):
        super().__init__('NOT', [node])

    def run(self, context):
        result = self.nodes[0].run(context)
        return self.Result.Success if result == self.Result.Failure else self.Result.Failure


