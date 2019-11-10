#
# wad.py, doom-net
#
# Created by Andrey Kolishchak on 07/04/17.
#
# based on https://gist.github.com/jasonsperske/42284303cf6a7ef19dc3
#
import struct
import re
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


class Wad(object):
    """Encapsulates the data found inside a WAD file"""

    def __init__(self, wadFile):
        """Each WAD files contains definitions for global attributes as well as map level attributes"""
        self.levels = []
        self.wad_format = 'DOOM'  # Assume DOOM format unless 'BEHAVIOR'
        with open(wadFile, "rb") as f:
            header_size = 12
            self.wad_type = f.read(4)[0]
            self.num_lumps = struct.unpack("<I", f.read(4))[0]
            data = f.read(struct.unpack("<I", f.read(4))[0] - header_size)

            current_level = Level(None)  # The first few records of a WAD are not associated with a level

            lump = f.read(16)  # Each offset is is part of a packet 16 bytes
            while len(lump) == 16:
                filepos = struct.unpack("<I", lump[0:4])[0] - header_size
                size = struct.unpack("<I", lump[4:8])[0]
                name = lump[8:16].decode('UTF-8').rstrip('\0')
                #print(name)
                if (re.match('E\dM\d|MAP\d\d', name)):
                    # Level nodes are named things like E1M1 or MAP01
                    if (current_level.is_valid()):
                        self.levels.append(current_level)

                    current_level = Level(name)
                elif name == 'BEHAVIOR':
                    # This node only appears in Hexen formated WADs
                    self.wad_format = 'HEXEN'
                else:
                    current_level.lumps[name] = data[filepos:filepos + size]

                lump = f.read(16)
            if (current_level.is_valid()):
                self.levels.append(current_level)

        for level in self.levels:
            level.load(self.wad_format)


class Level(object):
    """Represents a level inside a WAD which is a collection of lumps"""

    def __init__(self, name):
        self.name = name
        self.lumps = dict()
        self.vertices = []
        self.lower_left = None
        self.upper_right = None
        self.shift = None
        self.lines = []
        self.things = []
        self.sectors = []
        self.sides = []

    def is_valid(self):
        return self.name is not None and 'VERTEXES' in self.lumps and 'LINEDEFS' in self.lumps

    def normalize(self, point, padding=5):
        return (self.shift[0] + point[0] + padding, self.shift[1] + point[1] + padding)

    def load(self, wad_format):
        for vertex in packets_of_size(4, self.lumps['VERTEXES']):
            x, y = struct.unpack('<hh', vertex[0:4])
            self.vertices.append((x, y))

        self.lower_left = (min((v[0] for v in self.vertices)), min((v[1] for v in self.vertices)))
        self.upper_right = (max((v[0] for v in self.vertices)), max((v[1] for v in self.vertices)))

        self.shift = (0 - self.lower_left[0], 0 - self.lower_left[1])

        packet_size = 14
        for data in packets_of_size(packet_size, self.lumps['LINEDEFS']):
            self.lines.append(Line(data))

        packet_size = 10
        for data in packets_of_size(packet_size, self.lumps['THINGS']):
            self.things.append(Thing(data))

        packet_size = 26
        for data in packets_of_size(packet_size, self.lumps['SECTORS']):
            self.sectors.append(Sector(data))

        packet_size = 30
        for data in packets_of_size(packet_size, self.lumps['SIDEDEFS']):
            self.sides.append(Side(data))

    def save_svg(self):
        """ Scale the drawing to fit inside a 1024x1024 canvas (iPhones don't like really large SVGs even if they have the same detail) """
        import svgwrite
        view_box_size = self.normalize(self.upper_right, 10)
        if view_box_size[0] > view_box_size[1]:
            canvas_size = (1024, int(1024 * (float(view_box_size[1]) / view_box_size[0])))
        else:
            canvas_size = (int(1024 * (float(view_box_size[0]) / view_box_size[1])), 1024)

        dwg = svgwrite.Drawing(self.name + '.svg', profile='tiny', size=canvas_size,
                               viewBox=('0 0 %d %d' % view_box_size))
        for line in self.lines:
            a = self.normalize(self.vertices[line.a])
            b = self.normalize(self.vertices[line.b])
            if line.is_one_sided():
                dwg.add(dwg.line(a, b, stroke='#333', stroke_width=10))
            else:
                dwg.add(dwg.line(a, b, stroke='#999', stroke_width=3))

        dwg.save()

    def is_line_blocking(self, line):
        if line.is_blocking():
            return True

        return False
        '''
        # skip tagged sectors
        left_sector = self.sectors[self.sides[line.left_side].sector]
        right_sector = self.sectors[self.sides[line.right_side].sector]
        if left_sector.tag != 0 or right_sector.tag != 0:
            return False

        # blocking if floor levels different more than 16
        return abs(left_sector.floor_height - right_sector.floor_height) > 16
        '''

    def is_line_door(self, line):
        if line.is_door():
            return True

        left_sector = self.sectors[self.sides[line.left_side].sector]
        right_sector = self.sectors[self.sides[line.right_side].sector]
        if left_sector.tag != 0 or right_sector.tag != 0:
            return True

    def get_map(self):
        return LevelMap(self)


class Line(object):
    """Represents a Linedef inside a WAD"""

    def __init__(self, data):
        self.a, self.b, self.flags, self.special, self.sector_tag, self.left_side, self.right_side = \
            struct.unpack('<hhhhhhh', data)

    def is_one_sided(self):
        return self.left_side == -1 or self.right_side == -1

    def is_exit(self):
        return self.special in [11, 52, 51, 124, 197, 198]

    def is_blocking(self):
        return (self.flags & 0xc001) != 0

    def is_door(self):
        return self.special in [1, 26, 27, 28, 31, 32, 33, 34, 117, 118]
                                #29, 63, 4, 90, 103, 61, 2, 86, 50, 42, 3, 75, 16, 76, 46, 111, 114, 108]
                                #105, 112, 115, 109, 106, 113, 116, 110, 107, 135, 134, 133, 99, 137, 136]


class Thing(object):
    """Represents a Thing inside a WAD"""
    def __init__(self, data):
        self.x, self.y, self.angle, self.type, self.flags = struct.unpack('<hhHHH', data)

    def is_obstacle(self):
        return self.type in [57]


class Sector(object):
    """Represents a Sector inside a WAD"""
    def __init__(self, data):
        self.floor_height, _, _, _, _, _, self.tag = struct.unpack('<hhqqhHH', data)


class Side(object):
    """Represents a Sidedef inside a WAD"""
    def __init__(self, data):
        self.x, self.y, _, _, _, self.sector = struct.unpack('<hhqqqH', data)


class LevelMap(object):
    def __init__(self, level):
        vertices, lines, things = level.vertices, level.lines, level.things
        #
        # create a scaled map
        #
        scale = 10
        blocking_lines = [line for line in lines if level.is_line_blocking(line)]
        point_a = np.array([vertices[line.a] for line in blocking_lines]) // scale
        point_b = np.array([vertices[line.b] for line in blocking_lines]) // scale
        points = np.vstack([point_a, point_b])

        max_x, max_y = points.max(0)
        min_x, min_y = points.min(0)
        size_x = max_x - min_x + 1
        size_y = max_y - min_y + 1
        point_a -= np.array([min_x, min_y])
        point_b -= np.array([min_x, min_y])
        point_y, point_x = [], []
        for a, b in zip(point_a, point_b):
            y, x = skimage.draw.line(a[1], a[0], b[1], b[0])
            point_y.extend(y)
            point_x.extend(x)

        points = np.stack([point_y, point_x], axis=1)

        self.map = np.zeros(shape=(size_y, size_x), dtype=int)
        self.map[points[:, 0], points[:, 1]] = 1
        plt.imsave('level_map.png', self.map, cmap=cm.gray)

        # doors map
        door_lines = [line for line in lines if level.is_line_door(line)]
        if door_lines:
            point_a = np.array([vertices[line.a] for line in door_lines]) // scale
            point_b = np.array([vertices[line.b] for line in door_lines]) // scale

            point_a -= np.array([min_x, min_y])
            point_b -= np.array([min_x, min_y])
            point_y, point_x = [], []
            for a, b in zip(point_a, point_b):
                y, x = skimage.draw.line(a[1], a[0], b[1], b[0])
                point_y.extend(y)
                point_x.extend(x)

            points = np.stack([point_y, point_x], axis=1)

            self.map_doors = np.zeros(shape=(size_y, size_x), dtype=int)
            self.map_doors[points[:, 0], points[:, 1]] = 1
            map_doors = np.flip(self.map_doors, axis=0)
            plt.imsave('level_map_doors.png', map_doors, cmap=cm.gray)


        # save scales
        self.scale = 1/scale
        self.min_x = min_x
        self.min_y = min_y

        # add things
        for thing in things:
            if thing.is_obstacle():
                y, x = self.game_to_map(thing.y, thing.x)
                y -= 2
                x -= 2
                for r in range(5):
                    for c in range(5):
                        self.map[y+r, x+c] = 1

        self.vertices = vertices
        self.lines = lines
        self.things = things

        # create a distance map for exit
        exits = self.get_exits()
        self.exit_distance = None
        if exits:
            self.exit_distance = self.get_distance_map(*exits[0])

    def get_exits(self):
        exits = []
        for line in self.lines:
            if line.is_exit():
                x1, y1 = self.vertices[line.a]
                x2, y2 = self.vertices[line.b]
                exits.append(((y1+y2)/2, (x1+x2)/2))
        print('Exits: ', exits)
        return exits

    def game_to_map(self, y, x):
        y = int(y*self.scale - self.min_y)
        x = int(x*self.scale - self.min_x)
        return y, x

    def get_distance_map(self, y, x):
        y, x = self.game_to_map(y, x)
        distance_map = np.ndarray(shape=self.map.shape, dtype=float)
        distance_map.fill(-1)
        distance_map[y, x] = 0
        cells = [[y, x, 0]]
        while cells:
            cell_y, cell_x, distance = cells.pop(0)
            distance += 1
            for dy, dx in [[0, -1], [-1, 0], [0, 1], [1, 0]]:
                y = cell_y + dy
                x = cell_x + dx
                if 0 <= y < self.map.shape[0] and 0 <= x < self.map.shape[1] and \
                        self.map[y, x] == 0 and distance_map[y, x] == -1:
                    distance_map[y, x] = distance
                    cells.append([y, x, distance])

        for row in range(distance_map.shape[0]):
            for col in range(distance_map.shape[1]):
                if distance_map[row, col] == -1:
                    for dy, dx in [[0, -1], [-1, 0], [0, 1], [1, 0]]:
                        y = row + dy
                        x = col + dx
                        if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1] and distance_map[y, x] >= 0:
                            distance_map[y, x] = -2

        return distance_map

    def get_exit_distance(self, y, x):
        y, x = self.game_to_map(y, x)
        distance = self.exit_distance[y, x]
        return distance if distance >= 0 else self.exit_distance.max()

    def get_exit_heading(self, pose_y, pose_x):
        pose_y, pose_x = self.game_to_map(pose_y, pose_x)
        min_dist = self.exit_distance.max()
        heading_idx = 0
        for idx, (dy, dx) in enumerate([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]):
            y = pose_y + dy
            x = pose_x + dx
            if 0 <= y < self.exit_distance.shape[0] and 0 <= x < self.exit_distance.shape[1] and \
                    self.exit_distance[y, x] >= 0 and self.exit_distance[y, x] < min_dist:
                min_dist = self.exit_distance[y, x]
                heading_idx = idx

        heading = heading_idx * 45
        return heading

    def get_health(self):
        # TODO:
        return []

    def get_ammo(self):
        # TODO:
        return []

def packets_of_size(n, data):
    size = len(data)
    index = 0
    while index < size:
        yield data[index: index + n]
        index = index + n
    return


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        wad = Wad(sys.argv[1])
        for level in wad.levels:
            level.save_svg()
    else:
        print('You need to pass a WAD file as the only argument')