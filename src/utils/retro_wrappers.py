# This is an edited version of the Retro and Atari Wrappers from the Stable-Baselines package -- https://github.com/hill-a/stable-baselines
# It is used for the pre-processing of the frame data from the game environment
# Affecting actions, rewards and state information

import cv2
import numpy as np
import os
import gym
import retro

from collections import deque
from gym import spaces

cv2.ocl.setUseOpenCL(False)
os.environ.setdefault('PATH', '')


# Skipping frames to make the games non-deterministic
class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i==1:
                self.curac = ac
            if self.supports_want_render and i<self.n-1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done: break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)

# Wrap a gym-retro environment and make it use discrete
# actions for the Sonic game.
class SonicDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        self.actions_ = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],['DOWN', 'B'], ['B']]
        self._actions = []
        for action in self.actions_:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    # added for the no of actions
    def n_actions(self):
        return len(self.actions_)

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

    def re_convert(self, a):
        self._actions[a].copy()


# A replica of the SonicDiscretizer, altered to work with AirStriker
class AirstrikerDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(AirstrikerDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]

        # Actions changed from sonic
        self.actions_ = [['LEFT'], ['RIGHT'], ['B']]
        self._actions = []
        for action in self.actions_:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    # added for the no of actions
    def n_actions(self):
        return len(self.actions_)

    def action(self, a):
        return self._actions[a].copy()

    def re_convert(self, a):
        self._actions[a].copy()


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    # Bin reward to {+1, 0, -1} by its sign.
    def reward(self, reward):
        return np.sign(reward)

# Warp frames to 84x84 as done in the Nature paper and later work.
# If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
# observation should be warped.
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


# Stack k last frames - Returns lazy array, which is much more memory efficient
# See LazyFrames class below
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


# Create a retro environment with stochastic frame skipping
def make_retro(*, game, state=None, Stochastic_FrameSkip=True, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)

    if Stochastic_FrameSkip:
        env = StochasticFrameSkip(env, n=4, stickprob=0.25)

    return env


# Configure environment for retro games, using config similar to DeepMind-style Atari in wrap_deepmind
def wrap_deepmind_retro(env, scale=True, frame_stack=4):
    # Downscale the image and clip the rewards
    env = WarpFrame(env)
    env = ClipRewardEnv(env)

    # Stack frames on top of one another - default output (84, 84, 4)
    if frame_stack > 1:     env = FrameStack(env, frame_stack)
    if scale:               env = ScaledFloatFrame(env)

    return env
