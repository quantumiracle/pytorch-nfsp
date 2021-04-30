import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import random
import numpy as np
from collections import deque

from common.utils import epsilon_scheduler, update_target, print_log, load_model, save_model
from model import DQN, Policy
from storage import ParallelReplayBuffer, ReservoirBuffer

import gym
# import lasertag
import time, os
from tensorboardX import SummaryWriter

from common.utils import create_log_dir, print_args, set_global_seeds
from common.wrappers import wrap_pytorch, make_env
from arguments import get_args
from common.env import DummyVectorEnv, SubprocVectorEnv
from eq_solver import NashEquilibriaSolver, NashEquilibriumSolver

class ParallelNashAgent():
    def __init__(self, env, id, args):
        super(ParallelNashAgent, self).__init__()
        self.id = id
        self.env = env
        self.args = args
        self.current_model = DQN(env, args, Nash=True).to(args.device)
        self.target_model = DQN(env, args, Nash=True).to(args.device)
        update_target(self.current_model, self.target_model)

        if args.load_model and os.path.isfile(args.load_model):
            self.load_model(model_path)

        self.epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
        self.replay_buffer = ParallelReplayBuffer(args.buffer_size)
        self.rl_optimizer = optim.Adam(self.current_model.parameters(), lr=args.lr)

    def compute_nash(self, q_values, return_dist=False):
        """
        Return actions as Nash equilibrium of given payoff matrix, shape: [env, agent]
        """
        q_table = q_values.reshape(-1, self.env.action_space[0].n,  self.env.action_space[0].n)
        all_actions = []
        all_dists = []
        for qs in q_table:  # iterate over envs
            ne = NashEquilibriaSolver(qs)
            ne = ne[0]  # take the first Nash equilibria found
            actions = []
            all_dists.append(ne)
            for dist in ne:  # iterate over agents
                try:
                    sample_hist = np.random.multinomial(1, dist)
                except:
                    print(qs, ne)
                    print(dist)
                a = np.where(sample_hist>0)
                actions.append(a)
            # print(actions)
            all_actions.append(np.array(actions).reshape(-1))
        # print(all_actions)
        if return_dist:
            return all_dists
        else:
            return np.array(all_actions)

    def save_model(self, model_path):
        torch.save(self.current_model.state_dict(), model_path+f'/{self.id}_dqn')
        torch.save(self.target_model.state_dict(), model_path+f'/{self.id}_dqn_target')

    def load_model(self, model_path, eval=False, map_location=None):
        self.current_model.load_state_dict(torch.load(model_path+f'/{self.id}_dqn', map_location=map_location))
        self.target_model.load_state_dict(torch.load(model_path+f'/{self.id}_dqn_target', map_location=map_location))
        if eval:
            self.current_model.eval()
            self.target_model.eval()

def train(env, args, writer, model_path, num_agents=2):
    # agent_list = []
    # for i in range(num_agents):
    #     agent_list.append(ParallelNashAgent(env, i, args))  # [p0, p1]
    agent = ParallelNashAgent(env, 0, args)

    # Logging
    length_list = []
    reward_list = [[] for _ in range(num_agents)]
    rl_loss_list = [[] for _ in range(num_agents)]
    episode_reward = [0 for _ in range(num_agents)]
    tag_interval_length = 0
    prev_time = time.time()
    prev_frame = 1

    # Main Loop
    states =  env.reset()
    for frame_idx in range(1, args.max_frames + 1): # each step contains args.num_envs steps actually due to parallel envs
        # actions_ = []
        # for i in range(num_agents):
        #     epsilon = agent_list[i].epsilon_by_frame(frame_idx)
        #     try:
        #         state = states[:, i]  # obs: (env, agent, obs_dim)
        #     except:
        #         print('Number of envs needs to be larger than 1.')
        #     action = agent_list[i].current_model.act(torch.FloatTensor(state).to(args.device), epsilon)
        #     actions_.append(action)
        q_values = agent.current_model(torch.FloatTensor(states.reshape(states.shape[0], -1)).to(args.device)).detach().cpu().numpy() # concate states of all agents
        actions_ = agent.compute_nash(q_values)  
        assert num_agents == 2
        actions = [{"first_0": a0, "second_0": a1} for a0, a1 in zip(*actions_.T)] # a replicate of actions, actually the learnable agent is "second_0"
        print(actions)
        next_states, rewards, dones, infos = env.step(actions)
        done = [np.float32(d) for d in dones]

        # states (env, agent, state_dim) -> (env, agent*state_dim), similar for actions_, rewards take the positive one in two agents 
        samples = [[states[j].reshape(-1), actions_[j].reshape(-1), rewards[j, 0], next_states[j].reshape(-1), d] for j, d in enumerate(done) if not np.all(d)]
        agent.replay_buffer.push(samples) 
        
        # for i in range(num_agents):
        #     # filter out the samples of terminated env 
        #     samples = [[states[j, i], actions_[i][j], rewards[j, i], next_states[j, i], d] for j, d in enumerate(done) if not d]
        #     agent_list[i].replay_buffer.push(samples) 
        
        info = [list(i.values())[1] for i in infos]  # infos is a list of dicts (env) of dicts (agents)
        states = next_states

        # Logging
        for i in range(num_agents):
            episode_reward[i] += np.mean(rewards[:, i])  # mean over envs
        tag_interval_length += 1

        if np.any(done):  # TODO if use np.all(done), pettingzoo env will not provide obs for env after done
            length_list.append(tag_interval_length)
            tag_interval_length = 0

        # Episode done. Reset environment and clear logging records
        if np.any(done) or tag_interval_length >= args.max_tag_interval:
            states =  env.reset()  # p1_state=p2_state
            for i in range(num_agents):
                reward_list[i].append(episode_reward[i])
                writer.add_scalar(f"p{i}/episode_reward", episode_reward[i], frame_idx*args.num_envs)
            writer.add_scalar("data/tag_interval_length", tag_interval_length, frame_idx*args.num_envs)
            tag_interval_length = 0
            episode_reward = [0 for _ in range(num_agents)]

        if frame_idx % args.train_freq == 0:
            # for i in range(num_agents):
            if (len(agent.replay_buffer) > args.rl_start):
                # Update Best Response with Reinforcement Learning
                rl_loss = compute_rl_loss(agent, args)
                rl_loss_list[i].append(rl_loss.item())

                if frame_idx % args.max_tag_interval == 0:  # not log at every step
                    writer.add_scalar(f"p{i}/rl_loss", rl_loss.item(), frame_idx*args.num_envs)

        if frame_idx % args.update_target == 0:
            # for i in range(num_agents):
            #     update_target(agent_list[i].current_model, agent_list[i].target_model)
            update_target(agent.current_model, agent.target_model)

        # Logging and Saving models
        if frame_idx % args.evaluation_interval == 0:
            print(f"Frame: {frame_idx*args.num_envs}, Avg. Length: {np.mean(length_list):.1f}"+\
                ''.join([f", P{i} Avg. Reward: {np.mean(reward_list[i]):.3f}, P{i} Avg. RL Loss: {np.mean(rl_loss_list[i]):.3f}" for i in range(num_agents)]))
            reward_list = [[] for _ in range(num_agents)]
            rl_loss_list = [[] for _ in range(num_agents)]
            length_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()

            # for i in range(num_agents):
                # agent_list[i].save_model(model_path)
            agent.save_model(model_path)

        # Render if rendering argument is on
        if args.render:
            env.render()

    # for i in range(num_agents):
    #     agent_list[i].save_model(model_path)
    agent.save_model(model_path)


def compute_rl_loss(agent, args):
    current_model, target_model, replay_buffer, optimizer = agent.current_model, agent.target_model, agent.replay_buffer, agent.rl_optimizer
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
    weights = torch.ones(args.batch_size)
    # print(state.shape)
    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    # action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    # Q-Learning with target network
    q_values = current_model(state)
    target_next_q_values_ = target_model(next_state)
    target_next_q_values = target_next_q_values_.detach().cpu().numpy()
    # print(q_values.shape)

    action_dim = int(np.sqrt(q_values.shape[-1])) # for two-symmetric-agent case only
    action = torch.LongTensor([a[0]*action_dim+a[1] for a in action]).to(args.device)
    # print(action.shape)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    # next_q_value = target_next_q_values.max(1)[0]  # original one, get the maximum of target Q

    # compute CCE or NE
    # 1. NE
    nash_actions = agent.compute_nash(target_next_q_values, return_dist=True)  # get the mixed strategy Nash rather than specific actions
    target_next_q_values_ = target_next_q_values_.reshape(-1, action_dim, action_dim)
    nash_actions_  = torch.FloatTensor(nash_actions).to(args.device)
    next_q_value = torch.einsum('bk,bk->b', torch.einsum('bj,bjk->bk', nash_actions_[:, 0], target_next_q_values_), nash_actions_[:, 1])
    # 2. CCE

    expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

    # Huber Loss
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def test(env, args, model_path, num_agents=2): 
    # agent_list = []
    # for i in range(num_agents):
    #     agent = ParallelNashAgent(env, i, args)
    #     agent.load_model(model_path, eval=True, map_location='cuda:0')
    #     agent_list.append(agent)  # [p0, p1]
    agent = ParallelNashAgent(env, 0, args)
    agent.load_model(model_path, eval=True, map_location='cuda:0')  

    print('Load model from: ', model_path)

    reward_list = [[] for _ in range(num_agents)]
    length_list = []

    for _ in range(30):
        states = env.reset()
        episode_reward = [0 for _ in range(num_agents)]
        episode_length = 0
        t = 0
        while True:
            if args.render:
                env.render()
                # time.sleep(0.05)
            # actions = []
            # for i in range(num_agents):
            #     action = agent_list[i].current_model.act(torch.FloatTensor(states[i]).to(args.device), 0.)  # greedy action
            #     actions.append(action)
            q_values = agent.current_model(torch.FloatTensor(states.reshape(states.shape[0], -1)).to(args.device)).detach().cpu().numpy() # concate states of all agents
            actions = agent.compute_nash(q_values)  
            actions = {"first_0": actions[0], "second_0": actions[1]}  # a replicate of actions, actually the learnable agent is "second_0"
            next_states, reward, done, _ = env.step(actions)

            states = next_states
            for i in range(num_agents):
                episode_reward[i] += reward[i]
            episode_length += 1

            if done:
            # if done or t>=args.max_tag_interval:  # the pong game might get stuck after a while: https://github.com/PettingZoo-Team/PettingZoo/issues/357
                for i in range(num_agents):
                    reward_list[i].append(episode_reward[i])
                length_list.append(episode_length)
                break
            t += 1
    print("Test Result - Length {:.2f} ".format(np.mean(length_list))+\
        ''.join([f'P{i} Reward {np.mean(reward_list[i]):.2f}' for i in range(num_agents)]))

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret

def main():
    args = get_args()
    print_args(args)
    model_path = f'models/nash_dqn/{args.env}'
    os.makedirs(model_path, exist_ok=True)

    log_dir = create_log_dir(args)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)
    SEED = 721
    if args.num_envs == 1 or args.evaluate:
        env = make_env(args)  # "SlimeVolley-v0", "SlimeVolleyPixel-v0" 'Pong-ram-v0'
    else:
        VectorEnv = [DummyVectorEnv, SubprocVectorEnv][1]  # https://github.com/thu-ml/tianshou/blob/master/tianshou/env/venvs.py
        env = VectorEnv([lambda: make_env(args) for _ in range(args.num_envs)])
    print(env.observation_space, env.action_space)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        test(env, args, model_path)
        env.close()
        return

    train(env, args, writer, model_path)

    # writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
