import argparse
import gym
import multiprocessing
import tensorflow as tf
import threading

from doomenv import DoomEnv
from helper import *
from network import AC_Network
from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete
from time import sleep
from worker import Worker

max_episode_length = 300
gamma = .99  # discount rate for advantage estimation and reward discounting
s_size = 7056  # Observations are greyscale frames of 84 * 84 * 1
lr = 1e-4
load_model = False
env_name = "ppaquette/DoomBasic-v0"
model_path = "results/model"
frames_path = "results/frames"

if __name__ == "__main__":

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    env = gym.make(env_name)
    if "doom" in env_name.lower():
        env = SetResolution("160x120")(ToDiscrete("minimal")(env))

    a_size = env.action_space.n

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name="global_episodes", trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=lr)
        master_network = AC_Network(s_size, a_size, "global", None)
        num_workers = multiprocessing.cpu_count()
        workers = [
            Worker(env, i, s_size, a_size, trainer, model_path, global_episodes)
            for i in range(num_workers)
        ]
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print("Loading Model...")
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # Start the "work" process for each worker in a separate thread
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
