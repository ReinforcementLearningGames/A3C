from gym.wrappers import SkipWrapper
from gym import wrappers
from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete
from tensorpack.RL.gymenv import GymEnv


class DoomEnv(GymEnv):
    """ Wrapper for GymEnv that preprocesses for Doom
    """

    def __init__(self, name, resolution="160x120", commands="minimal", dumpdir=None, viz=False, auto_restart=True):
        """
        Args:
            name (str): the gym environment name.
            dumpdir (str): the directory to dump recordings to.
            viz (bool): whether to start visualization.
            auto_restart (bool): whether to restart after episode ends.
        """
        super().__init__(name, dumpdir, viz, auto_restart)
        self.gymenv = SetResolution(resolution)(ToDiscrete(commands)(self.gymenv))
