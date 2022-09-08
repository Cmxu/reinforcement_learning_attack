import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import sys
from rl_agent import training, ppo_training, DDPG, SAC, TD3, PPO
from data_util import toDeviceDataLoader, load_cifar, to_device
from model_util import VGG
from util import asr, geo_asr, show_attack, diff_affine, project_lp
from attack_sim import attack_simulation


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(43)

dataset_root = 'D:/datasets/'
cifar10_train, cifar10_val, cifar10_test = load_cifar(dataset_root)
train_loader, val_loader, test_loader = toDeviceDataLoader(cifar10_train, cifar10_val, cifar10_test, device = device)

mdl = to_device(VGG('VGG16'), device)
mdl.load_state_dict(torch.load('../models/torch_cifar_vgg.pth'))
mdl = mdl.eval()

env = attack_simulation(mdl = mdl, train_ds = cifar10_train, test_ds = cifar10_test, geometric_xi = 0.2, input_reduction = 'pca', in_pca_components = 512, action_reduction = 'geometric', device = device)

agent = TD3(env=env, ac_kwargs=dict(hidden_sizes=[256] * 2), gamma=0.99, num_test_episodes=5, max_ep_len=40)
training(agent = agent, dir = './tmp', steps_per_epoch=200, epochs=500, n_runs=1, start_steps=10000)