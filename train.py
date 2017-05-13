import argparse
import gym
import multiprocessing
import tensorflow as tf
import threading

from helper import *
from network import AC_Network
from time import sleep
from worker import Worker

model_path = "results/model"
frames_path = "results/frames"

parser = argparse.ArgumentParser()
parser.add_argument("--max_episode_length", type=int, default=300, required=False)
parser.add_argument("--gamma", type=float, default=0.99, required=False,
    help="Discount rate for advantage estimation and reward discounting")
parser.add_argument("--observation_dim", type=int, default=7056, required=False)
parser.add_argument("--lr", type=float, default=1e-4, required=False)
parser.add_argument("--load", action="store_true", required=False)
parser.add_argument("--env_name", type=str, default="ppaquette/DoomBasic-v0", required=False)
args = parser.parse_args()

if __name__ == "__main__":

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    env = gym.make(args.env_name)
    if "doom" in args.env_name.lower():
        env = wrap_doom(env)

    a_size = env.action_space.n

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name="global_episodes", trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=args.lr)
        master_network = AC_Network(args.observation_dim, a_size, "global", None)
        num_workers = multiprocessing.cpu_count()
        workers = [
            Worker(env, i, args.observation_dim, a_size, trainer, model_path, global_episodes)
            for i in range(num_workers)
        ]
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if args.load == True:
            print("Loading Model...")
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # Start the "work" process for each worker in a separate thread
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(args.max_episode_length, args.gamma, sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
