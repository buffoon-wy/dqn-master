import numpy as np
import torch
import os


class ReplayBuffer_normal:

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):

        """
        Constructor of ReplayBuffer class.
            
        :params:
            - buffer_capacity: capacity of memory for learning process
            - batch_size: size of batch sampling from memory for learning over them

        :return:
            None

        """


        self.batch_size = batch_size
        self.buffer_counter = 0

        self.buffer_capacity = buffer_capacity

        self.state_buffer = np.zeros((self.buffer_capacity, state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, ACTIONS_SIZE))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, STATE_SIZE))

    def save_buffer(self):

        """
        Function for saving current replay buffer into .csv file.
            
        :params:
            None

        :return:
            None

        """

        global SELECTED_MODEL

        np.savetxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/state_buffer.csv', self.state_buffer, delimiter=',')
        np.savetxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/action_buffer.csv', self.action_buffer, delimiter=',')
        np.savetxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/reward_buffer.csv', self.reward_buffer, delimiter=',')
        np.savetxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/next_state_buffer.csv', self.next_state_buffer, delimiter=',')

    def record(self, observation):  

        """
        Function for recording experience.
            
        :params:
            - observation: dictionary with keys:
                           - 'state': numpy array with shape (STATE_SIZE, ), containing current state, for whom best actions are calculated
                           - 'action': numpy array with shape (ACTIONS_SIZE, ), containing sampled actions
                           - 'reward': reward value for sampled actions
                           - 'next_state': numpy array with shape (STATE_SIZE, ), containing new/next state where we came, because of taken actions

        :return:
            None

        """

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = observation['state']
        self.action_buffer[index] = observation['action']
        self.reward_buffer[index] = observation['reward']
        self.next_state_buffer[index] = observation['next_state']

        self.buffer_counter += 1



class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

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
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.idx = end

