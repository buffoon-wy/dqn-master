from random import random
import torch
import numpy as np
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import utils
import csv
# define super paraments
FRAME_STACK = 4


def calculate_similarity(vect_1 ,vect_2):

    # 观测：余弦相似度 ，范围[-1~1]
    obs_dis = torch.cosine_similarity(torch.tensor(vect_1[0]), torch.tensor(vect_2[0]), dim=0).numpy()
    next_obs_dis = torch.cosine_similarity(torch.tensor(vect_1[3]), torch.tensor(vect_2[3]), dim=0).numpy()
    # 奖励：L1 损失 
    reward_dis = F.smooth_l1_loss(torch.tensor(vect_1[1]), torch.tensor(vect_2[1]), reduction='none').numpy()
    # 离散动作：相同为1，不同为0
    if vect_1[2] == vect_2[2]:
        action_dis = 1
    else:
        action_dis = 0

    # 计算相似度总分
    similarity_metric = obs_dis + reward_dis + action_dis + next_obs_dis 
    return similarity_metric


######################################################################
# Replay Buffer
 
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
            # this buffer is too large to be saved ,be care of the mounts 
            # self.obses[self.last_save:self.idx],
            # self.next_obses[self.last_save:self.idx],
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
            self.idx = end

    def similarity(self, number_choose):
        assert number_choose <= self.idx
        test_transition = []
        test_transition.append(self.features[number_choose])
        test_transition.append(self.rewards[number_choose])
        test_transition.append(self.actions[number_choose])
        test_transition.append(self.next_features[number_choose])
        
        for i in range(self.idx):
            cur_transition = []
            cur_transition.append(self.features[i])
            cur_transition.append(self.rewards[i])
            cur_transition.append(self.actions[i])
            cur_transition.append(self.next_features[i])
            similarity_grade = calculate_similarity(test_transition, cur_transition)
            self.sim_grade[i] = similarity_grade

        return similarity_grade


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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        return self.l2(x.view(-1, 512))

    def encoder(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.l1(x.view(x.size(0), -1)))  # output shape (512)

        return x


def main():
    # super paraments
    SEED = 1
    obs_shape = (FRAME_STACK, 84, 84)
    act_shape = 1
    replay_buffer_capacity =100000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed_everywhere(SEED)
    memory = ReplayBuffer(obs_shape, act_shape, replay_buffer_capacity, 64, device="cuda")
    policy_net = DQN(84, 84, 6).to(device)

    weights_path = "./weights1/policy_net_weights.pth"
    weight_model = torch.load(weights_path)
    policy_net.load_state_dict(weight_model)

    path_dir = "./buffer_4/"
    memory.load(path_dir)


    # transform the obs to feature vectors
    for i in range(memory.idx):
        obs = torch.tensor(memory.obses[i])
        obs = obs.unsqueeze(0).to(memory.device)
        memory.features[i] = policy_net.encoder(obs).squeeze(0).cpu().detach()
        next_obs = torch.tensor(memory.obses[i])
        next_obs = next_obs.unsqueeze(0).to(memory.device)
        memory.next_features[i] = policy_net.encoder(next_obs).squeeze(0).cpu().detach()

    memory.similarity(200)
    # memory.save_more('./buffer_more')
    # sim_grade = memory.sim_grade.max()
    # max_index = np.argmax(np.array(memory.sim_grade))
    # print(memory.sim_grade[max_index])
    # print(max_index)

    # save the action

    fo = open("action.csv", "w")
    header = ["seq", "act"]
    writer = csv.DictWriter(fo, header)
    writer.writeheader()
    action = memory.actions.tolist()
    
    for i in range(memory.idx):
        sim = str(action[i]).replace("[","")
        sim = sim.replace("]","")
        writer.writerow({"seq": i+1, "act":sim})

    # save the reward

    fo = open("reward.csv", "w")
    header = ["seq", "reward"]
    writer = csv.DictWriter(fo, header)
    writer.writeheader()
    rew = memory.rewards.tolist()
    for i in range(memory.idx):
        sim = str(rew[i]).replace("[","")
        sim = sim.replace("]","")
        writer.writerow({"seq": i+1, "reward":sim})

    # save the sim_grade

    fo = open("sim.csv", "w")
    header = ["seq", "sim_grade"]
    writer = csv.DictWriter(fo, header)
    writer.writeheader()
    rew = memory.sim_grade.tolist()
    for i in range(memory.idx):
        sim = str(rew[i]).replace("[","")
        sim = sim.replace("]","")
        writer.writerow({"seq": i+1, "sim_grade":sim})

    # save the image of the obs
    '''
        for i in range(memory.idx):
            raw_trans = memory.obses[i]    # shape: 4*84*84
            for j in range(4):
                gray_obs = raw_trans[j]
                path_img = "./img/" + str(i+1) + "_" +str(j)+ ".jpg"
                cv2.imwrite(path_img,gray_obs*255)
    '''

    print("LT SB")


if __name__ == '__main__':
    main()

