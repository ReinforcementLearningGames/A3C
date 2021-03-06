import argparse
import gym
import multiprocessing
import os
import tensorflow as tf
import threading

from utils.helper import *
from agent.network import AC_Network
from agent.worker import Worker
from gym import wrappers
from time import sleep

model_path = "results/model"
frames_path = "results/frames"

parser = argparse.ArgumentParser()
parser.add_argument("--max_episode_length", type=int, default=300, required=False)
parser.add_argument("--num_episodes", type=int, default=10, required=False)
parser.add_argument("--observation_dim", type=int, default=7056, required=False)
parser.add_argument("--env_name", type=str, default="ppaquette/DoomBasic-v0", required=False)
args = parser.parse_args()

if __name__ == "__main__":

    # Make model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Make gif directory
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    # Create Gym environment
    env = get_env(args.env_name)
    # Monitor environment and save videos
    env = wrappers.Monitor(env, "results/videos")
    # Get number of actions available in environment
    a_size = env.action_space.n

    # Use CPU
    with tf.device("/cpu:0"):
        # Create counter for number of episodes
        global_episodes = tf.Variable(0, dtype=tf.int32, name="global_episodes", trainable=False)
        # Instantiate actor critic network
        master_network = AC_Network(args.observation_dim, a_size, "global", None)
        # User Adam optimizer with LR = 0
        trainer = tf.train.AdamOptimizer(learning_rate=0)
        # Create worker who has instance of environment
        test_worker = Worker(env, 0, args.observation_dim, a_size, trainer, model_path, global_episodes)
        # Create model saver
        saver = tf.train.Saver()

    with tf.Session() as sess:
        # Coordinates threads
        coord = tf.train.Coordinator()
        # Load model
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Evaluate model for num_episodes
        for i in range(args.num_episodes):
            reward = test_worker.run(args.max_episode_length, sess)
            print("Reward: {}".format(round(reward, 2)))
