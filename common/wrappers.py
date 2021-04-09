import numpy as np
import gym
from gym import spaces
# from stable_baselines.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, MaxAndSkipEnv, WarpFrame
import slimevolleygym
from slimevolleygym import FrameStack # doesn't use Lazy Frames, easier to debug
import cv2

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return (np.swapaxes(observation[0], 2, 0), np.swapaxes(observation[1], 2, 0))
    

def wrap_pytorch(env):
    return ImageToPyTorch(env)

class NoopResetEnv(gym.Wrapper):
  def __init__(self, env, noop_max=30):
    """
    (from stable-baselines)
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    :param env: (Gym Environment) the environment to wrap
    :param noop_max: (int) the maximum value of no-ops to run
    """
    gym.Wrapper.__init__(self, env)
    self.noop_max = noop_max
    self.override_num_noops = None
    self.noop_action = 0
    assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    if self.override_num_noops is not None:
      noops = self.override_num_noops
    else:
      noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
    assert noops > 0
    obs = None
    for _ in range(noops):
      obs, _, done, _ = self.env.step(self.noop_action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs

  def step(self, action):
      return self.env.step(action)

class MaxAndSkipEnv(gym.Wrapper):
  def __init__(self, env, skip=4):
    """
    (from stable baselines)
    Return only every `skip`-th frame (frameskipping)
    :param env: (Gym Environment) the environment
    :param skip: (int) number of `skip`-th frame
    """
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=env.observation_space.dtype)
    self._skip = skip

  def step(self, action):
    """
    Step the environment with the given action
    Repeat action, sum reward, and max over last observations.
    :param action: ([int] or [float]) the action
    :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
    """
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2:
        self._obs_buffer[0] = obs
      if i == self._skip - 1:
        self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame
    # doesn't matter
    max_frame = self._obs_buffer.max(axis=0)
    return max_frame, total_reward, done, info

  def reset(self, **kwargs):
      return self.env.reset(**kwargs)

class WarpFrame(gym.ObservationWrapper):
  def __init__(self, env):
    """
    (from stable-baselines)
    Warp frames to 84x84 as done in the Nature paper and later work.
    :param env: (Gym Environment) the environment
    """
    gym.ObservationWrapper.__init__(self, env)
    self.width = 84
    self.height = 84
    self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                        dtype=env.observation_space.dtype)

  def observation(self, frame):
    """
    returns the current observation from a frame
    :param frame: ([int] or [float]) environment frame
    :return: ([int] or [float]) the observation
    """
    print("warp frame")
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]


## added for SlimVolley and PettingZoo 
import pettingzoo
import slimevolleygym  # https://github.com/hardmaru/slimevolleygym
import supersuit  # wrapper for pettingzoo envs

AtariEnvs = ['basketball_pong_v1', 'boxing_v1', 'combat_plane_v1', 'combat_tank_v1',
 'double_dunk_v2', 'entombed_competitive_v2', 'entombed_cooperative_v2', 'flag_capture_v1', 
 'foozpong_v1', 'ice_hockey_v1', 'joust_v2', 'mario_bros_v2', 'maze_craze_v2', 'othello_v2',
  'pong_v1', 'quadrapong_v2', 'space_invaders_v1', 'space_war_v1', 'surround_v1', 'tennis_v2', 
  'video_checkers_v3', 'volleyball_pong_v1', 'warlords_v2', 'wizard_of_wor_v2']

# import envs: multi-agent environments in PettingZoo Atari (both competitive and coorperative)
for env in AtariEnvs:   
    exec("from pettingzoo.atari import {}".format(env)) 

def make_env(env_name='boxing_v1', seed=1, obs_type='rgb_image'):
    '''https://www.pettingzoo.ml/atari'''
    if "slimevolley" in env_name or "SlimeVolley" in env_name:
        env = gym.make(env_name)
        env = SlimeVolleyWrapper(env)  # slimevolley to pettingzoo style

        # TODO apply these wrappers for pettingzoo style env, cannot use supersuit since SlimeVolley is neither pettingzoo nor gym env.
        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = WarpFrame(env)  # TODO maybe not use this since it's not an atari game
        # #env = ClipRewardEnv(env)
        # env = FrameStack(env, 4)
        env = NFSPPettingZooWrapper(env)  # pettingzoo to nfsp style 

    elif env_name in AtariEnvs: # PettingZoo envs
        env = eval(env_name).parallel_env(obs_type=obs_type)
        # print(env.action_spaces)

        if obs_type == 'rgb_image':
            # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
            # to deal with frame flickering
            env = supersuit.max_observation_v0(env, 2)

            # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
            env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

            # skip frames for faster processing and less control
            # to be compatable with gym, use frame_skip(env, (2,5))
            env = supersuit.frame_skip_v0(env, 4)

            # downscale observation for faster processing
            env = supersuit.resize_v0(env, 84, 84)

            # allow agent to see everything on the screen despite Atari's flickering screen problem
            env = supersuit.frame_stack_v1(env, 4)

        #   env = PettingZooWrapper(env)  # need to be put at the end
        
        # assign observation and action spaces
        env.observation_space = list(env.observation_spaces.values())[0]
        env.action_space = list(env.action_spaces.values())[0]
        env = NFSPPettingZooWrapper(env)

    else: # laserframe
        env = gym.make(env_name)
        env = wrap_pytorch(env)    

    env.seed(seed)
    return env

class SlimeVolleyWrapper():
    # action transformation of SlimeVolley 
    action_table = [[0, 0, 0], # NOOP
                    [1, 0, 0], # LEFT (forward)
                    [1, 0, 1], # UPLEFT (forward jump)
                    [0, 0, 1], # UP (jump)
                    [0, 1, 1], # UPRIGHT (backward jump)
                    [0, 1, 0]] # RIGHT (backward)


    def __init__(self, env):
        super(SlimeVolleyWrapper, self).__init__()
        self.env = env
        self.agents = ['first_0', 'second_0']
        self.observation_space = self.env.observation_space
        self.observation_spaces = {name: self.env.observation_space for name in self.agents}
        self.action_space = spaces.Discrete(len(self.action_table))
        self.action_spaces = {name: self.action_space for name in self.agents}


    def reset(self, observation=None):
        obs1 = self.env.reset()
        obs2 = obs1 # both sides always see the same initial observation.
        obs = {}
        obs[self.agents[0]] = obs1
        obs[self.agents[1]] = obs2
        return obs

    def seed(self, SEED):
        self.env.seed(SEED)

    def render(self,):
        self.env.render()

    def step(self, actions, against_baseline=False):
        obs, rewards, dones, infos = {},{},{},{}
        actions_ = [self.env.discreteToBox(a) for a in actions.values()]  # from discrete to multibinary action
        if against_baseline:
            # this is for validation: load a single policy as 'second_0' to play against the baseline agent (via self-play in 2015)
            obs2, reward, done, info = self.env.step(actions_[1]) # extra argument
            obs1 = obs2 
        else:
            # normal 2-player setting
            obs1, reward, done, info = self.env.step(*actions_) # extra argument
            obs2 = info['otherObs']

        obs[self.agents[0]] = obs1
        obs[self.agents[1]] = obs2
        rewards[self.agents[0]] = -reward
        rewards[self.agents[1]] = reward # the reward is for the learnable agent (second)
        dones[self.agents[0]] = done
        dones[self.agents[1]] = done
        infos[self.agents[0]] = info
        infos[self.agents[1]] = info

        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

class NFSPSlimeVolleyWrapper(SlimeVolleyWrapper):
    """ Wrap the SlimeVolley env to have a similar style as LaserFrame in NFSP """
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8) 
        fake_env = gym.make('Pong-v0')
        self.spec = fake_env.spec
        try:
            self.spec.id = env.env.spec.id
        except:
            pass
        fake_env.close()

    def observation_swapaxis(self, observation):
        return (np.swapaxes(observation[0], 2, 0), np.swapaxes(observation[1], 2, 0))
    
    def reset(self):
        obs_dict = super().reset()
        return self.observation_swapaxis(tuple(obs_dict.values()))

    def step(self, actions, against_baseline=False):
        obs, rewards, dones, infos = super().step(actions, against_baseline)
        o = self.observation_swapaxis(tuple(obs.values()))
        r = list(rewards.values())
        d = np.any(np.array(list(dones.values())))
        del obs,rewards, dones
        return o, r, d, infos

# class NFSPSlimeVolleyWrapper():
#     # action transformation of SlimeVolley 
#     action_table = [[0, 0, 0], # NOOP
#                     [1, 0, 0], # LEFT (forward)
#                     [1, 0, 1], # UPLEFT (forward jump)
#                     [0, 0, 1], # UP (jump)
#                     [0, 1, 1], # UPRIGHT (backward jump)
#                     [0, 1, 0]] # RIGHT (backward)


#     def __init__(self, env):
#         super(NFSPSlimeVolleyWrapper, self).__init__()
#         self.env = env
#         self.agents = ['first_0', 'second_0']
#         old_shape = self.env.observation_space.shape
#         self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8) 
#         self.observation_spaces = {name: self.env.observation_space for name in self.agents}
#         self.action_space = spaces.Discrete(len(self.action_table))
#         self.action_spaces = {name: self.action_space for name in self.agents}
#         fake_env = gym.make('Pong-v0')
#         self.spec = fake_env.spec
#         try:
#             self.spec.id = env.env.spec.id
#         except:
#             pass
#         fake_env.close()

#     def reset(self, observation=None):
#         obs1 = self.env.reset()
#         obs2 = obs1 # both sides always see the same initial observation.
#         return (np.swapaxes(obs1, 2, 0), np.swapaxes(obs2, 2, 0))

#     def seed(self, SEED):
#         self.env.seed(SEED)

#     def render(self,):
#         self.env.render()

#     def step(self, actions, against_baseline=False):
#         actions_ = [self.env.discreteToBox(a) for a in actions.values()]  # from discrete to multibinary action

#         if against_baseline:
#             # this is for validation: load a single policy as 'second_0' to play against the baseline agent (via self-play in 2015)
#             obs2, reward, done, info = self.env.step(actions_[1]) # extra argument
#             obs1 = obs2 
#         else:
#             # normal 2-player setting
#             obs1, reward, done, info = self.env.step(*actions_) # extra argument
#             obs2 = info['otherObs']

#         obs = (np.swapaxes(obs1, 2, 0), np.swapaxes(obs2, 2, 0))
#         rewards = [-reward, reward]

#         return obs, rewards, done, info

#     def close(self):
#         self.env.close()


class NFSPPettingZooWrapper():
    """ Wrap the PettingZoo envs to have a similar style as LaserFrame in NFSP """
    def __init__(self, env):
        super(NFSPPettingZooWrapper, self).__init__()
        self.env = env
        old_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)
        self.action_space = env.action_space
        fake_env = gym.make('Pong-v0')
        self.spec = fake_env.spec
        try:
            self.spec.id = env.env.spec.id
        except:
            pass
        fake_env.close()

    def observation_swapaxis(self, observation):
        return (np.swapaxes(observation[0], 2, 0), np.swapaxes(observation[1], 2, 0))
    
    def reset(self):
        obs_dict = self.env.reset()
        return self.observation_swapaxis(tuple(obs_dict.values()))

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        o = self.observation_swapaxis(tuple(obs.values()))
        r = list(rewards.values())
        d = np.any(np.array(list(dones.values())))
        info = list(infos.values())
        del obs,rewards, dones, infos
        return o, r, d, info

    def seed(self, SEED):
        self.env.seed(SEED)

    def render(self,):
        self.env.render()

    def close(self):
        self.env.close()
