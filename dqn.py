# -*- coding: utf-8 -*-


from ast import Pass
import imp
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

import utils
from utils import set_seed_everywhere
from utils import FrameStack
from repaly_buffer import ReplayBuffer


import time
env = gym.make('SpaceInvaders-v0').unwrapped


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Input extraction

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize((84, 84), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor()])


def get_screen():
    # Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


######################################################################
# Training

# 参数和网络初始化
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 10



episode_durations = []


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def random_start(skip_steps=30, m=4):
    env.reset()
    state_queue = deque([], maxlen=m)
    next_state_queue = deque([], maxlen=m)
    done = False
    for i in range(skip_steps):
        if (i+1) <= m:
            state_queue.append(get_screen())
        elif m < (i + 1) <= 2*m:
            next_state_queue.append(get_screen())
        else:
            state_queue.append(next_state_queue[0])
            next_state_queue.append(get_screen())

        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            break
    return done, state_queue, next_state_queue


######################################################################
# Start Training

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--task_name', default='highway')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--skip', default=3, type=int)
    parser.add_argument('--num_train_steps', default=100000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    set_seed_everywhere(seed=args.seed)
    env = FrameStack(env, k=args.frame_stack)

    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape
    obs_shape = (3, screen_height, screen_width)
    act_shape = env.action_space.n

    replay_buffer = ReplayBuffer(obs_shape, act_shape, args.replay_buffer_capacity, args.batch_size, device)

    policy_net = DQN(screen_height, screen_width, act_shape).to(device)
    target_net = DQN(screen_height, screen_width, act_shape).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())


    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        if done:
            # evaluate agent periodically
            if episode % args.eval_freq == 0:               
                # evaluate(eval_env, agent, video, args.num_eval_episodes, L, step)
                if args.save_model:
                    pass
                if args.save_buffer:
                    pass

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 5 if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, curr_reward, reward, next_obs, done_bool)
        np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

        obs = next_obs
        episode_step += 1

        
        
        memory.push(state, action, next_state, m_reward)

        state = next_state
        optimize_model()

        if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), 'weights/policy_net_weights_{0}.pth'.format(i_episode))


    print('Complete')
    env.close()
    torch.save(policy_net.state_dict(), 'weights/policy_net_weights.pth')


if __name__ == '__main__':
    main()
