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
    total_reward = 0

    for _ in range(100):
        state = env.reset()
        terminal = False
        episode_reward = 0
        steps = 0
        while not terminal:
            env.render()
            action = player.act(state, actions)
            state, reward, terminal, info = env.step(action)
            steps += 1
            episode_reward += reward
            if terminal:
                print("episode terminated after %d steps with a total reward of %g" % (steps, episode_reward))
                total_reward += episode_reward

    print("Avg reward: %g" % float(total_reward) / 100.0)
    
