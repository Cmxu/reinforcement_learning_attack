import torch
from util import project_lp, diff_affine
import torch.nn.functional as F
import numpy as np

def fgsm(x, y, k, norm = np.inf, xi = 1e-1, step_size = 1e-1, device = torch.device('cuda:0')):
    v = torch.zeros_like(x, device = device, requires_grad = True)
    loss = F.cross_entropy(k(x + v), y)
    loss.backward()
    return project_lp(step_size * torch.sign(v.grad), norm = norm, xi = xi)

def pgd(x, y, k, norm = np.inf, xi = 1e-1, step_size = 1e-2, epochs = 40, random_restart = 4, device = torch.device('cuda:0')):
    batch_size = x.shape[0]
    max_loss = F.cross_entropy(k(x), y)
    max_X = torch.zeros_like(x)
    random_delta = torch.rand(size = (batch_size * random_restart, *x.shape[1:]), device = device) - 0.5
    random_delta = project_lp(random_delta, norm = norm, xi = xi, exact = True, device = device)
    x = x.repeat(random_restart, 1, 1, 1)
    y = y.repeat(random_restart)
    for j in range(epochs):
        v = torch.zeros_like(random_delta, device = device, requires_grad = True)
        loss = F.cross_entropy(k(x + random_delta + v), y)
        loss.backward()
        pert = step_size * torch.sign(v.grad)#torch.mean(v.grad)
        random_delta = project_lp(random_delta + pert, norm = norm, xi = xi)
    _,idx = torch.max(F.cross_entropy(k(x + random_delta), y, reduction = 'none').reshape(random_restart, batch_size), axis = 0)
    return random_delta[idx * batch_size + torch.arange(batch_size, dtype = torch.int64, device = device)]

def geometric_attack(x, y, k, xi = 2e-1, step_size = 1e-2, epochs = 40, device = torch.device('cuda:0')):
    batch_size = x.shape[0]
    v = torch.zeros(batch_size, 2, 3, device = device)
    for i in range(epochs):
        v_ = torch.zeros(batch_size, 2, 3, device = device, requires_grad = True)
        x_ = diff_affine(x, v + v_)
        loss = F.cross_entropy(k(x_), y)
        loss.backward()
        pert = step_size * torch.sign(v_.grad)
        v = project_lp(v + pert, norm = np.inf, xi = xi)
    return v

def pgd_pca(x, y, k, pca, norm = np.inf, xi = 1e-1, step_size = 1e-2, epochs = 40, random_restart = 4, device = torch.device('cuda:0'), inflate = True):
    batch_size = x.shape[0]
    max_loss = F.cross_entropy(k(x), y)
    max_X = torch.zeros_like(x)
    random_delta = torch.rand(size = (batch_size * random_restart, *x.shape[1:]), device = device) - 0.5
    random_delta = project_lp(random_delta, norm = norm, xi = xi, exact = True, device = device)
    random_delta = pca.reduce(random_delta)
    x = x.repeat(random_restart, 1, 1, 1)
    y = y.repeat(random_restart)
    for j in range(epochs):
        v = torch.zeros_like(random_delta, device = device, requires_grad = True)
        loss = F.cross_entropy(k(x + pca.inflate(random_delta + v)), y)
        loss.backward()
        pert = step_size * torch.sign(v.grad)#torch.mean(v.grad)
        random_delta = pca.reduce(project_lp(pca.inflate(random_delta + pert), norm = norm, xi = xi))
    _,idx = torch.max(F.cross_entropy(k(x + pca.inflate(random_delta)), y, reduction = 'none').reshape(random_restart, batch_size), axis = 0)
    v = random_delta[idx * batch_size + torch.arange(batch_size, dtype = torch.int64, device = device)]
    return project_lp(pca.inflate(v), norm = norm, xi = xi) if inflate else v
    
   