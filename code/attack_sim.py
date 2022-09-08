import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from data_util import toDeviceDataLoader
from util import project_lp, diff_affine

class torch_pca():
    def __init__(self, data, n_components=512, device = torch.device('cuda:0')):
        self.device = device
        self.n_components = n_components
        transformer = PCA(n_components=self.n_components)
        data = torch.utils.data.DataLoader(data, batch_size=len(data))
        data = next(iter(data))[0].numpy()
        self.data_shape = data.shape[1:]
        transformer.fit(data.reshape(data.shape[0], -1))
        self.explained_variance = sum(transformer.explained_variance_ratio_)
        self.components_ = torch.Tensor(transformer.components_).to(device)
        self.pca_mean_ = torch.Tensor(transformer.mean_).to(device)

    def reduce(self, x):
        x = x.reshape(x.shape[0], -1)
        return torch.matmul(x - self.pca_mean_, self.components_.T)

    def inflate(self, x):
        inf = torch.matmul(x, self.components_) + self.pca_mean_
        return inf.reshape(inf.shape[0], *self.data_shape)

class attack_simulation():
    def __init__(self, mdl, train_ds, test_ds, norm=np.inf, xi=1e-1, step_size=1e-2,
                 input_reduction=None, action_reduction=None, device = torch.device('cuda:0'), **kwargs):
        self.device = device
        self.step_size = 1e-2
        self.xi = 1e-1
        self.norm = norm
        self.mdl = mdl
        if input_reduction is None:
            transform = lambda x: x
        elif input_reduction == 'pca':
            if 'in_pca_components' in kwargs:
                self.in_pca = torch_pca(train_ds, n_components=kwargs['in_pca_components'])
            else:
                self.in_pca = torch_pca(train_ds)
            self.in_transform = self.in_pca.reduce
        elif input_reduction == 'nn':
            raise Exception('{} reduction is not implemented'.format(input_reduction))
        else:
            raise Exception('{} reduction is not implemented'.format(input_reduction))
        self.train_dataset, self.test_dataset = toDeviceDataLoader(train_ds, test_ds, batch_size = 1, device = self.device, transform = self.in_transform)
        self.data_shape = torch.tensor(next(iter(self.train_dataset))[1].shape)

        self.action_reduction = action_reduction
        if self.action_reduction is None:
            self.action_shape = self.data_shape
        elif self.action_reduction == 'geometric':
            self.action_shape = torch.tensor([2, 3])
            self.geometric_xi = kwargs.get('geometric_xi', 0.2)
        elif self.action_reduction == 'patch':
            self.patch_size = kwargs.get('patch_size', 5)
            self.action_shape = torch.tensor([1, (3 * self.patch_size ** 2) + 2])
        elif self.action_reduction == 'pca':
            if 'out_pca_components' in kwargs:
                self.out_pca = torch_pca(train_ds, n_components=kwargs['out_pca_components'])
            elif 'in_pca_components' in kwargs:
                self.out_pca = self.in_pca
            else:
                self.out_pca = torch_pca(train_ds)
            self.out_transform = self.out_pca.inflate
            self.action_shape = torch.tensor([1, self.out_pca.n_components])
        else:
            raise Exception('{} reduction is not implemented'.format(self.action_reduction))
        out = self.start()
        if isinstance(out, (list, tuple)):
            self.state_shape = [torch.tensor(x.shape) for x in out]
        else:
            self.state_shape = torch.tensor(out.shape)
        self.reset_current_action()

        self.train_iterator = iter(self.train_dataset)
        self.test_iterator = iter(self.test_dataset)

    def reset_current_action(self):
        self.current_action = torch.zeros(*self.action_shape, device=self.device)

    def random_action(self):
        random_move = torch.zeros(*self.action_shape, device = self.device).uniform(-1, 1)
        return random_move

    def start(self):
        return self.reset()

    def test(self):
        self.reset_current_action()
        try:
            self.xt, self.x, self.y = next(self.test_iterator)
        except:
            self.train_iterator = iter(self.test_dataset)
            self.xt, self.x, self.y = next(self.test_iterator)
        self.x_base = torch.clone(self.x)
        self.current_loss = F.cross_entropy(self.mdl(self.x), self.y)
        return self.state_space_output()[0]
        
    def reset(self):
        self.reset_current_action()
        try:
            self.xt, self.x, self.y = next(self.train_iterator)
        except:
            self.train_iterator = iter(self.train_dataset)
            self.xt, self.x, self.y = next(self.train_iterator)
        self.x_base = torch.clone(self.x)
        self.current_loss = F.cross_entropy(self.mdl(self.x), self.y)
        return self.state_space_output()[0]

    def state_space_output(self):
        # return self.xt, self.mdl(self.x), self.y
        with torch.no_grad():
            pred = self.mdl(self.x)
            new_loss = F.cross_entropy(pred, self.y)
            difference = new_loss - self.current_loss
            self.current_loss = new_loss
        return self.xt, difference.reshape(1), (torch.argmax(pred) != self.y).reshape(1)

    def step(self, move):
        assert torch.all(torch.tensor(move.shape) == self.action_shape)
        move = move * self.step_size
        if self.action_reduction is None:
            self.current_action = project_lp(self.current_action + move, norm=self.norm, xi=self.xi)
            self.x = self.x_base + self.current_action
        elif self.action_reduction == 'geometric':
            self.current_action = project_lp(self.current_action + move, norm=np.inf, xi=self.geometric_xi)
            self.x = diff_affine(self.x_base, self.current_action)
        elif self.action_reduction == 'patch':
            self.current_action[:, :-2] = project_lp(self.current_action[:, :-2] + move[:, :-2], norm=self.norm,
                                                     xi=self.xi)
            self.current_action[:, -2:] = project_lp(self.current_action[:, -2:] + move[:, -2:], norm=np.inf, xi=1)
            patch_x = int((self.current_action[:, -2] + 1) / 2 * (self.data_shape[2] - self.patch_size))
            patch_y = int((self.current_action[:, -1] + 1) / 2 * (self.data_shape[3] - self.patch_size))
            self.x = self.x_base
            self.x[:, :, patch_x:patch_x + self.patch_size, patch_y:patch_y + self.patch_size] += move[:, :-2].reshape(
                3, self.patch_size, self.patch_size)
        elif self.action_reduction == 'pca':
            action_inf = self.out_pca.inflate(self.current_action + move)
            action_inf = project_lp(action_inf, norm=self.norm, xi=self.xi)
            self.current_action = self.out_pca.reduce(action_inf)
            self.x = self.x_base + action_inf
        else:
            raise Exception('{} reduction is not implemented'.format(self.action_reduction))
        self.xt = self.in_transform(self.x)
        return self.state_space_output()