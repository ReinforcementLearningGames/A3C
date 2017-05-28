import gym
import numpy as np
import random
import scipy.misc
import tensorflow as tf

from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete


def get_env(env_name):
    """ Makes gym environment.
    """
    env = gym.make(env_name)
    if "doom" in env_name.lower():
        env = wrap_doom(env)
    return env

def wrap_doom(env, resolution="640x480", actions="minimal"):
    """ Set resolution and make actions discrete for Doom.
    """
    return SetResolution(resolution)(ToDiscrete(actions)(env))


def process_frame(frame):
    """ Processes screen image to produce grayscale, resized image.
    """
    s = np.mean(frame, axis=-1)
    s = scipy.misc.imresize(s, [84, 84])
    s = np.reshape(s, [np.prod(s.shape)]) / 255.0
    return s


def discount(x, gamma):
    """ Discounting function used to calculate discounted returns.
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def normalized_columns_initializer(std=1.0):
    """ Used to initialize weights for policy and value output layers.
    """
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def update_target_graph(from_scope, to_scope):
    """ Copies one set of variables to another.
        Used to set worker network parameters to those of global network.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def make_gif(images, fname, duration=2):
    """ Creates gifs to be saved of the training episode for use in the Control Center.
    """
    import moviepy.editor as mpy

    def make_frame(t):
        """ Gets frame at timestep.
        """
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        return x.astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration, verbose=False)
