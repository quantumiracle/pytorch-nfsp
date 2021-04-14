import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import random
import numpy as np
from collections import deque

from common.utils import epsilon_scheduler, update_target, print_log, load_model, save_model
from model import DQN, Policy
from storage import ReplayBuffer, ReservoirBuffer

import gym
import lasertag
import time, os
from tensorboardX import SummaryWriter

from common.utils import create_log_dir, print_args, set_global_seeds
from common.wrappers import wrap_pytorch, make_env
from arguments import get_args
from test import test  # TODO

def train(env, args, writer, model_path):
    # RL Model for Player 1
    p1_current_model = DQN(env, args).to(args.device)
    p1_target_model = DQN(env, args).to(args.device)
    update_target(p1_current_model, p1_target_model)

    if args.load_model and os.path.isfile(args.load_model):
        p1_current_model.load_state_dict(torch.load(model_path+'/dqn'))
        p1_target_model.load_state_dict(torch.load(model_path+'/dqn_target'))

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)

    # Replay Buffer for Reinforcement Learning - Best Response
    p1_replay_buffer = ReplayBuffer(args.buffer_size)

    # Deque data structure for multi-step learning
    p1_state_deque = deque(maxlen=args.multi_step)
    p1_reward_deque = deque(maxlen=args.multi_step)
    p1_action_deque = deque(maxlen=args.multi_step)

    # RL Optimizer for Player 1, 2
    p1_rl_optimizer = optim.Adam(p1_current_model.parameters(), lr=args.lr)

    # Logging
    length_list = []
    p1_reward_list, p1_rl_loss_list = [], []
    p1_episode_reward = 0
    tag_interval_length = 0
    prev_time = time.time()
    prev_frame = 1

    # Main Loop
    (p1_state, p2_state) =  env.reset()  # p1_state=p2_state

    for frame_idx in range(1, args.max_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        p1_action = p1_current_model.act(torch.FloatTensor(p1_state).to(args.device), epsilon)

        actions = {"first_0": p1_action, "second_0": p1_action}  # a replicate of actions, actually the learnable agent is "second_0"
            
        next_states, rewards, dones, infos = env.step(actions, against_baseline=True)

        p1_next_state = next_states[1]  # the second one is learnable
        reward = rewards[1]
        done = dones
        info = infos[1]
        
        # Save current state, reward, action to deque for multi-step learning
        p1_state_deque.append(p1_state)
        p1_reward_deque.append(reward)
        p1_action_deque.append(p1_action)

        # Store (state, action, reward, next_state) to Replay Buffer for Reinforcement Learning
        if len(p1_state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(p1_reward_deque, args.gamma)
            n_state = p1_state_deque[0]
            n_action = p1_action_deque[0]
            p1_replay_buffer.push(n_state, n_action, n_reward, p1_next_state, np.float32(done))

        p1_state = p1_next_state

        # Logging
        p1_episode_reward += reward
        tag_interval_length += 1

        if done:
            length_list.append(tag_interval_length)
            tag_interval_length = 0
            
        # Episode done. Reset environment and clear logging records
        if done or tag_interval_length >= args.max_tag_interval:
            (p1_state, p2_state) =  env.reset()  # p1_state=p2_state
            p1_reward_list.append(p1_episode_reward)
            writer.add_scalar("p1/episode_reward", p1_episode_reward, frame_idx)
            writer.add_scalar("data/tag_interval_length", tag_interval_length, frame_idx)
            p1_episode_reward, tag_interval_length = 0, 0
            p1_state_deque.clear()
            p1_reward_deque.clear()
            p1_action_deque.clear()

        if (len(p1_replay_buffer) > args.rl_start and
            frame_idx % args.train_freq == 0):

            # Update Best Response with Reinforcement Learning
            p1_rl_loss = compute_rl_loss(p1_current_model, p1_target_model, p1_replay_buffer, p1_rl_optimizer, args)
            p1_rl_loss_list.append(p1_rl_loss.item())
            writer.add_scalar("p1/rl_loss", p1_rl_loss.item(), frame_idx)

        if frame_idx % args.update_target == 0:
            update_target(p1_current_model, p1_target_model)


        # Logging and Saving models
        if frame_idx % args.evaluation_interval == 0:
            print(f"Frame: {frame_idx}, Avg. Reward: {np.mean(p1_reward_list):.3f}, Avg. RL Loss: {np.mean(p1_rl_loss_list):.3f}, Avg. Length: {np.mean(length_list):.1f}")
            p1_reward_list.clear(), length_list.clear()
            p1_rl_loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()

            torch.save(p1_current_model.state_dict(), model_path+'/dqn')
            torch.save(p1_target_model.state_dict(), model_path+'/dqn_target')

        # Render if rendering argument is on
        if args.render:
            env.render()

    torch.save(p1_current_model.state_dict(), model_path+'/dqn')
    torch.save(p1_target_model.state_dict(), model_path+'/dqn_target')


def compute_rl_loss(current_model, target_model, replay_buffer, optimizer, args):
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
    weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    # Q-Learning with target network
    q_values = current_model(state)
    target_next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.max(1)[0]
    expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

    # Huber Loss
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def test(env, args, model_path): 
    p1_current_model = DQN(env, args).to(args.device)
    p1_current_model.eval()
    print('Load model from: ', model_path)
    p1_current_model.load_state_dict(torch.load(model_path+'/dqn', map_location='cuda:0'))

    p1_reward_list = []
    length_list = []

    for _ in range(30):
        (p1_state, p2_state) = env.reset()
        p1_episode_reward = 0
        p2_episode_reward = 0
        episode_length = 0
        while True:
            if args.render:
                env.render()
                # time.sleep(0.05)
            p1_action = p1_current_model.act(torch.FloatTensor(p1_state).to(args.device), 0.)  # greedy action
            actions = {"first_0": p1_action, "second_0": p1_action}  # a replicate of actions, actually the learnable agent is "second_0"
            (p1_next_state, p2_next_state), reward, done, _ = env.step(actions, against_baseline=True)

            (p1_state, p2_state) = (p1_next_state, p2_next_state)
            p1_episode_reward += reward[0]
            episode_length += 1

            if done:
                p1_reward_list.append(p1_episode_reward)
                length_list.append(episode_length)
                break
    
    print("Test Result - Length {:.2f} Reward {:.2f}".format(
        np.mean(length_list), np.mean(p1_reward_list)))
    

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret

def main():
    args = get_args()
    print_args(args)
    model_path = f'models/train_dqn_against_baseline/{args.env}'
    os.makedirs(model_path, exist_ok=True)

    log_dir = create_log_dir(args)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)
    SEED = 721
    env = make_env(args.env)  # "SlimeVolley-v0", "SlimeVolleyPixel-v0" 'Pong-ram-v0'

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
