import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from agent.network import AC_Network
from utils.helper import *

from random import choice
from time import sleep
from time import time


class Worker():
    def __init__(self, env, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("results/train_" + str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        #The Below code is related to setting up the Doom environment
        self.actions = [i for i in range(a_size)]
        #End Doom set-up
        self.env = env

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {
            self.local_AC.target_v: discounted_rewards,
            self.local_AC.inputs: np.vstack(observations),
            self.local_AC.actions: actions,
            self.local_AC.advantages: advantages,
            self.local_AC.state_in[0]: rnn_state[0],
            self.local_AC.state_in[1]: rnn_state[1]
        }
        v_l, p_l, e_l, g_n, v_n, _ = sess.run(
            [
                self.local_AC.value_loss, self.local_AC.policy_loss, self.local_AC.entropy, self.local_AC.grad_norms,
                self.local_AC.var_norms, self.local_AC.apply_grads
            ],
            feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def _get_feed_dict(self, state, rnn_state):
        return {
                self.local_AC.inputs: [state],
                self.local_AC.state_in[0]: rnn_state[0],
                self.local_AC.state_in[1]: rnn_state[1]
        }

    def _choose_action(self, sess, state, rnn_state):
        action_distribution, value, rnn_state = sess.run(
            [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
             self._get_feed_dict(state, rnn_state))

        action = np.random.choice(self.actions, p=action_distribution[0])
        return action, value, rnn_state

    def _step(self, action):
        next_state, reward, terminal, _ = self.env.step(action)
        reward /= 100.0
        return next_state, reward, terminal

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                in_terminal_state = False

                state = self.env.reset()
                episode_frames.append(state)
                state = process_frame(state)
                rnn_state = self.local_AC.state_init

                while not in_terminal_state:
                    #Take an action using probabilities from policy network output.
                    a, v, rnn_state = self._choose_action(sess, state, rnn_state)
                    next_state, reward, in_terminal_state = self._step(a)

                    if not in_terminal_state:
                        episode_frames.append(next_state)
                        next_state = process_frame(next_state)
                    else:
                        next_state = state

                    episode_buffer.append([state, a, reward, next_state, in_terminal_state, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += reward
                    state = next_state
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer
                           ) == 30 and not in_terminal_state and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, self._get_feed_dict(state, rnn_state))[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                v_l = p_l = e_l = g_n = v_n = 0.0
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 10 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)

                        make_gif(
                            images,
                            'results/frames/image' + str(episode_count) + '.gif',
                            duration=len(images) * time_per_step,
                            true_image=True,
                            salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
