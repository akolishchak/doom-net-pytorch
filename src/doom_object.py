#
# doom_instance.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#

class DoomObject:

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
        'ZombieMan'                # Former human trooper
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

    obstacle = [
        'Column',                  # Mini Tech Light
        'BurningBarrel',           # Barrel Fire
        'ExplosiveBarrel',         # Exploding Barrel(Doom)
        #'TechLamp',                # Large Tech Lamp
        'TechLamp2',               # Small Tech Lamp
        'TechPillar'               # Tech Column
    ]

    shot = [
        'Rocket'
    ]

    @staticmethod
    def get_id(label):
        if label.object_name in DoomObject.enemy:
            return 0  # enemy
        elif label.object_name in DoomObject.shot:
            return 1  # shot
        elif label.object_name in DoomObject.obstacle:
            return 2  # obstacle
        elif label.object_name in DoomObject.health:
            return 3  # health
        elif label.object_name in DoomObject.ammo:
            return 4  # ammo

        return -1  # unknown object