# -*- coding: utf-8 -*-


from inspect import stack
import queue
from sys import maxsize
from threading import current_thread
import turtle
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import os
import cv2 as cv
env = gym.make('SpaceInvaders-v0').unwrapped
import datetime

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数和网络初始化
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 10
FRAME_STACK = 1

######################################################################
# Replay Memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, action_shape), dtype=np.int)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
       
        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)
        print("save buffer %d_%d.pt",self.last_save, self.idx)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end



######################################################################
# DQN algorithm

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.l1 = nn.Linear(linear_input_size, 512)
        self.l2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        return self.l2(x.view(-1, 512))


######################################################################
# Input extraction

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize((84, 84), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor()])


def get_screen():
    # Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array')
    # cv.imshow("11",screen)
    # date = datetime.datetime.now()
    # str_date = str(date)
    # path = "./img/" + str_date +".jpg"
    # cv.imwrite(path, screen)
    # # cv.waitKey(1000)
    screen = screen.transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


######################################################################
# Training


init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

obs_shape = (FRAME_STACK, screen_height, screen_width)
act_shape = 1
replay_buffer_capacity =100000

memory = ReplayBuffer(obs_shape, act_shape, replay_buffer_capacity, BATCH_SIZE, device)

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


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
    if memory.idx < BATCH_SIZE:
        return
    obs, action, reward, next_obs, _ = memory.sample()
    batch_next_state = []
    for i in range(BATCH_SIZE):  
        a = next_obs[i] 
        b = a.unsqueeze(0)
        batch_next_state.append(b)
    next_state_new = tuple(batch_next_state)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_new)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in next_state_new if s is not None])


    state_action_values = policy_net(obs).gather(1, action)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    new_reward = reward.squeeze(1)
    expected_state_action_values = (next_state_values * GAMMA) + new_reward

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def random_start(skip_steps=30, m=FRAME_STACK):
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

num_episodes = 10000
m = FRAME_STACK
store_flag = True

for i_episode in range(num_episodes):
    # Initialize the environment and state
    done, state_queue, next_state_queue = random_start()
    if done:
        continue

    state = torch.cat(tuple(state_queue), dim=1)
    current_reward = 0

    for t in count():
        reward = 0
        m_reward = 0
        # 每m帧完成一次action
        action = select_action(state)
        # store replay buffer
        if store_flag:

            if memory.idx == 0:
                    date = datetime.datetime.now()
                    print(str(date))
            if memory.idx %20000 == 0:
                dir_buffer = "./buffer/"
                print("saving !!!")
                memory.save(dir_buffer)
                
            if memory.idx ==80001:
                date = datetime.datetime.now()
                print(str(date))
                store_flag = False

        for i in range(m):
            _, reward, done, _ = env.step(action.item())
            if not done:
                next_state_queue.append(get_screen())
            else:
                break
            m_reward += reward

        if not done:
            next_state = torch.cat(tuple(next_state_queue), dim=1)
        else:
            # game over! punishment!!
            next_state = None
            m_reward = -150

        m_reward = torch.tensor([m_reward], device=device)

        if done:
           pass
        else:       
            memory.add(state.cpu().numpy(), action.cpu().numpy(), m_reward.cpu().numpy(), next_state.cpu().numpy(), done)
            
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
