import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

######################################################################
# DQN algorithm

# import torch
# import torch.nn.functional as F

# vec1 = torch.FloatTensor([1, 2, 3, 4])
# vec2 = torch.FloatTensor([5, 6, 7, 8])

# cos_sim = F.cosine_similarity(vec1, vec2, dim=0)
# print(cos_sim) 

# import numpy as np
# vec1 = np.array([1, 2, 3, 4])
# vec2 = np.array([5, 6, 7, 8])

# cos_sim = vec1.dot(vec2) / np.linalg.norm(vec1) * np.linalg.norm(vec2)
# print(cos_sim)

# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# vec1 = np.array([1, 2, 3, 4])
# vec2 = np.array([-5, -6, -7, -8])

# cos_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))
# print(cos_sim[0][0])
# a = [1,2,3 , 4]
# a = np.array(a)
# b = a.max()
# print(b)

# import csv

# fo = open("sim_grade.csv", "w")

# header = ["seq", "value"]

# writer = csv.DictWriter(fo, header)
# writer.writeheader()
# a= 1
# b =2
# writer.writerow({"seq": a, "value":b})
# #关闭文件
# fo.close()


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, feature=512):
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
        self.features = np.empty((capacity, feature), dtype=np.float32)
        self.next_features = np.empty((capacity, feature), dtype=np.float32)
        self.sim_grade = np.empty((capacity, 1), dtype=np.float32)
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

    # def obs2trans(self, x_i, length):
    #     assert self.idx == length
    #     for i in range(self.idx):
    #         trans = torch.tensor(self.obses[i])
    #         trans = trans.unsqueeze(0).to(self.device)
    #         self.transitions[i] = self.encoder(self.obses[i]).squeeze(0)

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

    def save_more(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
            self.features[self.last_save:self.idx],
            self.next_features[self.last_save:self.idx],
            self.sim_grade[self.last_save:self.idx]
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
            self.not_dones[start:end] = payload[4]
            self.features[start:end] = payload[5]
            self.next_features[start:end] = payload[6]
            self.sim_grade[start:end] = payload[7]
            self.idx = end

    def load_more(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.actions[start:end] = payload[0]
            self.rewards[start:end] = payload[1]
            self.not_dones[start:end] = payload[2]
            self.features[start:end] = payload[3]
            self.next_features[start:end] = payload[4]
            self.sim_grade[start:end] = payload[5]
            self.idx = end



FRAME_STACK = 4
obs_shape = (FRAME_STACK, 84, 84)
act_shape = 1
replay_buffer_capacity =100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

memory = ReplayBuffer(obs_shape, act_shape, replay_buffer_capacity, 64, device="cuda")
path_dir = "./buffer_more/"
memory.load_more(path_dir)
raw_trans = memory.obses[9999]    # shape: 1*84*84
print("ok")