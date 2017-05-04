import gym
import argparse
from gym.wrappers import Monitor
from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete

RESOLUTION = "200X150"
COMMANDS = "minimal"
AUTO_RESTART = True

class RandoCommando(object):
"""
If he gets the job done, it's totally by accident.
"""
    
    def __init__(self):
        super()

    def act(self, state, actions):
        return actions.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='doom environment to run', default='ppaquette/DoomBasic-v0')
    args = parser.parse_args()

    env = gym.make(args.env)
    player = RandoCommando()
    actions = env.action_space

    while True:
        state = env.reset()
        terminal = False
        total_reward = 0
        steps = 0
        while not terminal:
            env.render()
            action = player.act(state, actions)
            state, reward, terminal, info = env.step(action)
            steps += 1
            total_reward += 1
            if terminal:
                print("episode terminated after %d steps with a total reward of %g" % (steps, total_reward))
