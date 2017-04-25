import numpy as np

"""
A wrapper for atari environment that keeps a grayscale frame buffer that is updated at every step.
"""

class FrameBufferWrapper(object):
    
    def __init__(self, env, frame_depth):
        self.env = env
        self.frame_depth = frame_depth
        self.frame_buffer = np.zeros(self._get_buffer_dimensions())
        
    def _get_buffer_dimensions(self):
        return self.env.reset().shape[0:2] + (self.frame_depth,)
        
    def step(self, action):
        next_frame, reward, terminal, info = self.env.step(action)
        self.frame_buffer = np.roll(self.frame_buffer, 1)
        self.frame_buffer[:,:,0] = self._to_grayscale(next_frame)
        return self.frame_buffer, reward, terminal, info
        
    def reset(self):
        next_frame = self.env.reset()
        self.frame_buffer[:,:,0] = self._to_grayscale(next_frame)
        return self.frame_buffer
        
    def _to_grayscale(self, img):
        return np.mean(img, axis=2)
        
