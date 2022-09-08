from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
from numpy import linalg as LA
import time
import argparse
import json
import torch.nn as nn
import itertools
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
#from spinup_copy import mpi_avg, mpi_statistics_scalar, num_procs, setup_pytorch_for_mpi, sync_params, mpi_avg_grads
import scipy.signal
import os

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float32, device='cuda:0')
        self.obs2_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float32, device='cuda:0')
        self.act_buf = torch.zeros(combined_shape(size, act_dim), dtype=torch.float32, device='cuda:0')
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device='cuda:0')
        self.done_buf = torch.zeros(size, dtype=torch.float32, device='cuda:0')
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float32, device='cuda:0')
        self.act_buf = torch.zeros(combined_shape(size, act_dim), dtype=torch.float32, device='cuda:0')
        self.adv_buf = torch.zeros(size, dtype=torch.float32, device='cuda:0')
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device='cuda:0')
        self.ret_buf = torch.zeros(size, dtype=torch.float32, device='cuda:0')
        self.val_buf = torch.zeros(size, dtype=torch.float32, device='cuda:0')
        self.logp_buf = torch.zeros(size, dtype=torch.float32, device='cuda:0')
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat(self.rew_buf[path_slice], last_val)
        vals = torch.cat(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers).to(torch.device('cuda'))

class DDPGActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = (torch.as_tensor(act_limit, dtype=torch.float32)).to(torch.device('cuda'))

    def forward(self, obs):
        return self.act_limit * self.pi(obs)

class DDPGQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class DDPGActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        act_limit = 1

        # build policy and value functions
        self.pi = DDPGActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = DDPGQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.device = torch.device('cuda')

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs)

class DDPG(object):
    def __init__(self, env, ac_kwargs=dict(), replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3,
                 q_lr=1e-3, batch_size=100, act_noise=0.1, num_test_episodes=10, max_ep_len=1000):
        self.name = 'ddpg'
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.act_noise = act_noise
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.ac_kwargs = ac_kwargs
        self.env = env

        self.obs_dim = self.env.state_shape[1].item()
        self.act_dim = torch.prod(env.action_shape).item()
        self.act_limit = 1
        self.ac = DDPGActorCritic(self.obs_dim, self.act_dim, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)
        self.pi_lr = pi_lr
        self.q_lr = q_lr


        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)
        self.replay_size = replay_size

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        return loss_q

    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.to(self.ac.device)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.to(self.ac.device)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o):
        noise_scale = self.act_noise
        a = self.ac.act(o)
        a += (noise_scale * torch.randn(self.act_dim)).to(device='cuda:0')
        return torch.clip(a, -self.act_limit, self.act_limit)

    def get_action_test(self, o):
        a = self.ac.act(o)
        return torch.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        ep_rets = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.env.test(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                test_a = self.get_action_test(o)
                test_a = test_a.reshape(*self.env.action_shape)
                o, r, d = self.env.step(test_a)
                ep_ret += r
                ep_len += 1
            ep_rets.append(ep_ret)
        return ep_rets

    def reset(self):
        self.ac = DDPGActorCritic(self.obs_dim, self.act_dim, **self.ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=self.q_lr)

    def random_action(self):
        return torch.zeros(self.act_dim, device='cuda:0').uniform_(-1, 1)


class TD3Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = (torch.as_tensor(act_limit, dtype=torch.float32)).to(torch.device('cuda'))

    def forward(self, obs):
        return self.act_limit * self.pi(obs)

class TD3QFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class TD3ActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        act_limit = 1

        # build policy and value functions
        self.pi = TD3Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = TD3QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = TD3QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.device = torch.device('cuda')

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs)

class TD3(object):
    def __init__(self, env, ac_kwargs=dict(), replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000):
        self.name = 'td3'
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.ac_kwargs = ac_kwargs
        self.env = env
        self.obs_dim = self.env.state_shape[1].item()
        self.act_dim = torch.prod(env.action_shape).item()
        self.act_limit = 1
        self.ac = TD3ActorCritic(self.obs_dim, self.act_dim, **ac_kwargs)
        self.ac_kwargs = ac_kwargs
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)
        self.replay_size = replay_size

    def random_action(self):
        return torch.zeros(self.act_dim, device='cuda:0').uniform_(-1, 1)

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    def compute_loss_pi(self, data):
        o = data['obs']
        q1_pi = self.ac.q1(o, self.ac.pi(o))
        return -q1_pi.mean()

    def update(self, data, timer):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.to(self.ac.device)
        loss_q.backward()
        self.q_optimizer.step()

        # Possibly update pi and target networks
        if timer % self.policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.to(self.ac.device)
            loss_pi.backward()
            self.pi_optimizer.step()

            for p in self.q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o):
        noise_scale = self.act_noise
        a = self.ac.act(o)
        a += noise_scale * torch.randn(self.act_dim).to(device='cuda:0')
        return torch.clip(a, -self.act_limit, self.act_limit)

    def get_action_test(self, o):
        a = self.ac.act(o)
        return torch.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        ep_rets = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.env.test(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                test_a = self.get_action_test(o)
                test_a = test_a.reshape(*self.env.action_shape)
                o, r, d = self.env.step(test_a)
                ep_ret += r
                ep_len += 1
            ep_rets.append(ep_ret)
        return ep_rets

    def reset(self):
        self.ac = TD3ActorCritic(self.obs_dim, self.act_dim, **self.ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.q_lr)
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SACActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim).to(torch.device('cuda'))
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim).to(torch.device('cuda'))
        self.act_limit = (torch.as_tensor(act_limit, dtype=torch.float32)).to(torch.device('cuda'))

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None
        if logp_pi != None:
            logp_pi.to(torch.device('cuda'))
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class SACQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class SACActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        act_limit = 1

        # build policy and value functions
        self.pi = SACActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = SACQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = SACQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.device = torch.device('cuda')

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

class SAC(object):
    def __init__(self, env, ac_kwargs=dict(), replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, num_test_episodes=10, max_ep_len=1000):
        self.name = 'sac'
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.lr =lr
        self.alpha = alpha
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.ac_kwargs = ac_kwargs
        self.env = env
        self.obs_dim = self.env.state_shape[1].item()
        self.act_dim = torch.prod(env.action_shape).item()
        self.act_limit = 1
        self.ac = SACActorCritic(self.obs_dim, self.act_dim, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)
        self.replay_size = replay_size
        self.ac_kwargs = ac_kwargs

    def random_action(self):
        return torch.zeros(self.act_dim, device='cuda:0').uniform_(-1, 1)

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        return loss_pi

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.to(self.ac.device)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.to(self.ac.device)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), False)

    def get_action_test(self, o):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), True)

    def test_agent(self):
        ep_rets = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.env.test(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                test_a = self.get_action_test(o)
                test_a = test_a.reshape(*self.env.action_shape)
                o, r, d = self.env.step(test_a)
                ep_ret += r
                ep_len += 1
            ep_rets.append(ep_ret)
        return ep_rets

    def reset(self):
        self.ac = SACActorCritic(self.obs_dim, self.act_dim, **self.ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class PPOCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class PPOGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std)).to(torch.device('cuda'))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std).to(torch.device('cuda'))
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

class PPOCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.

class PPOActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        self.pi = PPOGaussianActor(obs_dim, act_dim, hidden_sizes, activation)


        # build value function
        self.v = PPOCritic(obs_dim, hidden_sizes, activation)
        self.device = torch.device('cuda')

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]

class PPO(object):
    def __init__(self, env, ac_kwargs=dict(), seed=0, steps_per_epoch=4000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
        target_kl=0.01, num_test_episodes = 10, max_ep_len = 1000):

        setup_pytorch_for_mpi()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.name = 'ppo'
        self.ac_kwargs = ac_kwargs
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len

        self.env = env

        # Instantiate environment
        self.obs_dim = self.env.state_shape[1].item()
        self.act_dim = torch.prod(env.action_shape).item()
        self.act_limit = 1
        # Create actor-critic module
        self.ac = PPOActorCritic(self.obs_dim, self.act_dim, **ac_kwargs)

        # Sync params across processes
        sync_params(self.ac)

        # Set up experience buffer
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, gamma, lam)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self):
        data = self.buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.to(self.ac.device)
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
            self.pi_optimizer.step()


        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()

    def reset(self):
        self.ac = PPOActorCritic(self.env.observation_space, self.env.action_space)
        sync_params(self.ac)
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, self.gamma, self.lam)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

    def test(self):
        ep_rets = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.env.test(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.env.step(self.get_action_test(o).reshape(*self.env.action_shape))
                ep_ret += r
                ep_len += 1
            ep_rets.append(ep_ret)
        return ep_rets

    def get_action(self, obs):
        return self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.ac.device))[0]

    def get_action_test(self, obs):
        return self.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(self.ac.device))[0]

def training(agent, dir, steps_per_epoch=4000, epochs=100, batch_size=100, start_steps=10000,
         update_after=1000, update_every=50, max_ep_len=1000, n_runs=1):

    total_steps = steps_per_epoch * epochs

    score_log = []
    for i in range(n_runs):
        print('run', i)
        score = []
        o, ep_len = agent.env.reset(), 0
        for t in range(total_steps):
            if t > start_steps:
                a = agent.get_action(o)
            else:
                a = agent.random_action()


            # Step the env
            o2, r, d = agent.env.step(a.reshape(*agent.env.action_shape))

            ep_len += 1

            d = False if ep_len == max_ep_len else d

            agent.replay_buffer.store(o, a, r, o2, d)

            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                o, ep_ret, ep_len, ep_ret_clean = agent.env.reset(), 0, 0, 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = agent.replay_buffer.sample_batch(batch_size)
                    if agent.name == 'td3':
                        agent.update(data=batch, timer = j)
                    elif agent.name == 'ddpg' or 'sac':
                        agent.update(data=batch)

            # End of epoch handling
            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                # Test the performance of the deterministic version of the agent.

                ep_rets = agent.test_agent()
                score.append(ep_rets)
                print(epoch, [a.item() for a in ep_rets])
        score_log.append(score)
        agent.reset()
    data = dict()
    data['score'] = score_log
    with open(os.path.join(dir, 'outputs.json'), 'w') as f:
        f.write(json.dumps(data))


def ppo_training(agent, dir, steps_per_epoch=4000, epochs=200, n_runs=1, max_ep_len = 1000):
    score_log = []
    for i in range(n_runs):
        print('run', i)
        score = []
        o, ep_len = agent.env.reset(), 0
        for epoch in range(epochs):
            for t in range(steps_per_epoch):
                a, v, logp = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(agent.ac.device))
                next_o, r, d = agent.env.step(a.reshape(*agent.env.action_shape))

                ep_len += 1

                # save and log
                agent.buf.store(o, a, r, v, logp)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = agent.ac.step(torch.as_tensor(o, dtype=torch.float32).to(agent.ac.device))
                    else:
                        v = 0
                    agent.buf.finish_path(v)
                    o, ep_ret, ep_ret_clean, ep_len = agent.env.reset(), 0, 0, 0

            # Perform PPO update!
            agent.update()
            ep_rets = agent.test()
            score.append(ep_rets)
            print(epoch, ep_rets)
        score_log.append(score)
        agent.reset()
    data = dict()
    data['score'] = score_log
    with open(os.path.join(dir,'outputs.json'), 'w') as f:
        f.write(json.dumps(data))