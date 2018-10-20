#
# doom_object.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import math
import numpy as np


class DoomObject:

    class Type:
        UNKNOWN = -1
        AGENT = -2
        ENEMY = 0
        BULLET = 1
        OBSTACLE = 2
        HEALTH = 3
        AMMO = 4
        WEAPON = 5
        EXIT = 6
        EXIT_SIGN = 7
        DOOR = 8
        WALLS = 9
        MAX = 10
    '''
    class Type:
        UNKNOWN = -1
        AGENT = -2
        ENEMY = 1
        HEALTH = 2
        AMMO = 3
        EXIT = 4
        EXIT_SIGN = 5
        DOOR = 6
        WALLS = 7
        MAX = 8
    '''

    TYPE = 0
    X = 1
    Y = 2
    Z = 3
    HEADING = 4
    VELOCITY_X = 5
    VELOCITY_Y = 6

    @staticmethod
    def get_pose(type, x, y, z=0, heading=0, velocity_x=0, velociy_y=0):
        return np.array([type, x, y, z, heading, velocity_x, velociy_y], dtype=np.float32)

    ammo = [
        'Backpack',     # Backpack (Increase carrying capacity)
        'Cell',         # Cell
        'CellPack',     # Cell Pack
        'Clip',         # Ammo Clip
        'ClipBox',      # Box of Bullets
        'RocketAmmo',   # Rocket
        'RocketBox',    # Box of Rockets
        'Shell',        # 4 Shells
        'ShellBox'      # Box of Shells
    ]

    enemy = [
        'DoomPlayer',
        'Arachnotron',             # Arachnotron
        'Archvile',                # Arch-vile
        'BaronOfHell',             # Baron of Hell
        'HellKnight',              # Hell knight
        'Cacodemon',               # Cacodemon
        'Cyberdemon',              # Cyberdemon
        'Demon',                   # Demon
        'Spectre',                 # Partially invisible demon
        'ChaingunGuy',             # Former human commando
        'DoomImp',                 # Imp
        'Fatso',                   # Mancubus
        'LostSoul',                # Lost soul
        'PainElemental',           # Pain elemental
        'Revenant',                # Revenant
        'ShotgunGuy',              # Former human sergeant
        'SpiderMastermind',        # Spider mastermind
        'WolfensteinSS',           # Wolfenstein soldier
        'ZombieMan',                # Former human trooper
        'Zombieman'  # Former human trooper
    ]

    health = [
        'ArmorBonus',              # Armor Helmet
        'Berserk',                 # Berserk Pack (Full Health+Super Strength)
        'BlueArmor',               # Heavy Armor
        'BlurSphere',              # Partial Invisibility
        'GreenArmor',              # Light Armor
        'HealthBonus',             # Health Potion
        'InvulnerabilitySphere',   # Invulnerability
        'Medikit',                 # Medikit(+25 Health)
        'Megasphere',              # Megasphere (+200 Health/Armor)
        'RadSuit',                 # Radiation Suit
        'Soulsphere',              # Soul Sphere (+100 Health)
        'Stimpack'                 # Stimpack(+10 Health)
    ]

    weapon = [
        'Chainsaw',
        'Pistol',
        'Shotgun',
        'SuperShotgun',
        'Chaingun',
        'RocketLauncher',
        'PlasmaRifle',
        'BFG9000'
    ]

    obstacle = [
        #'Column',                  # Mini Tech Light
        'BurningBarrel',           # Barrel Fire
        'ExplosiveBarrel',         # Exploding Barrel(Doom)
        #'TechLamp',                # Large Tech Lamp
        #'TechLamp2',               # Small Tech Lamp
        #'TechPillar'               # Tech Column
    ]

    shot = [
        'Rocket'
    ]

    exit = [
        'Exit'
    ]

    exit_sign = [
        'ExitSign'
    ]

    door = [
        'Door'
    ]

    @staticmethod
    def get_id(label):
        object_type = DoomObject.Type.UNKNOWN

        if label.object_name in DoomObject.enemy:
            object_type = DoomObject.Type.ENEMY
        elif label.object_name in DoomObject.shot:
            object_type = DoomObject.Type.BULLET
        elif label.object_name in DoomObject.obstacle:
            object_type = DoomObject.Type.OBSTACLE
        elif label.object_name in DoomObject.health:
            object_type = DoomObject.Type.HEALTH
        elif label.object_name in DoomObject.ammo:
            object_type = DoomObject.Type.AMMO
        elif label.object_name in DoomObject.weapon:
            object_type = DoomObject.Type.WEAPON
        elif label.object_name in DoomObject.exit:
            object_type = DoomObject.Type.EXIT
        elif label.object_name in DoomObject.exit_sign:
            object_type = DoomObject.Type.EXIT_SIGN
        elif label.object_name in DoomObject.door:
            object_type = DoomObject.Type.DOOR

        return object_type
    '''
    @staticmethod
    def get_id(label):
        object_type = DoomObject.Type.UNKNOWN

        if label.object_name in DoomObject.enemy:
            object_type = DoomObject.Type.ENEMY
        elif label.object_name in DoomObject.health:
            object_type = DoomObject.Type.HEALTH
        elif label.object_name in DoomObject.ammo:
            object_type = DoomObject.Type.AMMO
        elif label.object_name in DoomObject.weapon:
            object_type = DoomObject.Type.WEAPON
        elif label.object_name in DoomObject.exit:
            object_type = DoomObject.Type.EXIT
        elif label.object_name in DoomObject.exit_sign:
            object_type = DoomObject.Type.EXIT_SIGN
        elif label.object_name in DoomObject.door:
            object_type = DoomObject.Type.DOOR

        return object_type
'''
