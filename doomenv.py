import gym
from gym.wrappers import SkipWrapper
from gym import wrappers
from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete
from tensorpack.RL.gymenv import GymEnv


class DoomEnv(GymEnv):
    """ Wrapper for GymEnv that preprocesses for Doom
    """

    def __init__(self, name, resolution="200x150", commands="minimal", dumpdir=None, viz=False, auto_restart=True):
        """
        Args:
            name (str): the gym environment name.
            dumpdir (str): the directory to dump recordings to.
            viz (bool): whether to start visualization.
            auto_restart (bool): whether to restart after episode ends.
        """
        super().__init__(name, dumpdir, viz, auto_restart)

        if dumpdir:
            self.gymenv = gym.make(name)
            self.gymenv = wrappers.Monitor(self.gymenv, dumpdir, force=True)

        self.gymenv = SetResolution(resolution)(ToDiscrete(commands)(self.gymenv))
        self.gymenv.reset()
