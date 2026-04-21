import math
import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn

# Updated import path
from weak_annotation.utils.data import apply_sliding_window

class PHuberCrossEntropy(nn.Module):
    def __init__(self, tau: float = 10, weight = None, label_smoothing: float = 0) -> None:
        super().__init__()
        self.tau = tau
        self.label_smoothing = label_smoothing
        self.prob_thresh = 1 / self.tau
        self.boundary_term = math.log(self.tau) + 1
        self.softmax = nn.Softmax(dim=1)
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert 0 <= self.label_smoothing < 1
        with torch.no_grad():
            smoothed_labels = self.label_smoothing / (input.size(1) - 1)* torch.ones_like(input)
            smoothed_labels.scatter_(1, target.data.unsqueeze(1).type(torch.int64), 1-self.label_smoothing)
            target = smoothed_labels
        p = torch.sum(self.softmax(input) * target, dim=1)
        w = torch.tile(self.weight, (len(p), 1))
        w = [torch.arange(w.shape[0]), target]
        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        if self.weight is not None:
            loss[clip] = (-self.tau * p[clip] + self.boundary_term) * w
            loss[~clip] = (-torch.log(p[~clip])) * w
        else:
            loss[clip] = -self.tau * p[clip] + self.boundary_term
            loss[~clip] = -torch.log(p[~clip])
        return loss

class PHuberGeneralizedCrossEntropy(nn.Module):
    def __init__(self, q: float = 0.7, tau: float = 10, weight = None, label_smoothing: float = 0) -> None:
        super().__init__()
        self.q = q
        self.tau = tau
        self.label_smoothing = label_smoothing
        self.weight = None
        self.prob_thresh = tau ** (1 / (q - 1))
        self.boundary_term = tau * self.prob_thresh + (1 - self.prob_thresh ** q) / q
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert 0 <= self.label_smoothing < 1
        p = torch.sum(self.softmax(input) * target, dim=1)
        w = torch.tile(self.weight, (len(p), 1))
        w = torch.sum(w * target, dim=1)
        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        if self.weight is not None:
            loss[clip] = (-self.tau * p[clip] + self.boundary_term) * w[clip]
            loss[~clip] = ((1 - p[~clip] ** self.q) / self.q) * w[~clip]
        else:
            loss[clip] = -self.tau * p[clip] + self.boundary_term
            loss[~clip] = (1 - p[~clip] ** self.q) / self.q
        return loss

class InertialDataset(Dataset):
    def __init__(self, data, window_size, window_overlap, classes, weights=None):
        if weights is not None:
            self.ids, self.features, labels, weights = apply_sliding_window(data, window_size, window_overlap, weights)
        else:
            self.ids, self.features, labels = apply_sliding_window(data, window_size, window_overlap, weights)
            weights = np.ones_like(labels)
        self.classes = classes
        labels_onehot = np.eye(self.classes)[labels]
        self.labels = labels_onehot
        self.weights = labels_onehot * weights[:, None]
        self.prob_vectors = np.zeros((len(self.labels), self.classes))
        self.channels = self.features.shape[2]
        self.window_size = window_size

    def __len__(self):
        return len(self.features)
    
    def update_labels(self, labels):
        self.labels = labels

    def __getitem__(self, index):
        return (
            self.ids[index].astype(np.uint8),
            self.features[index, :, :].astype(np.float32),
            self.labels[index].astype(np.float32),
            self.weights[index].astype(np.float32),
            self.prob_vectors[index].astype(np.float32)
        )

def init_weights(network, weight_init):
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            if weight_init == 'normal': nn.init.normal_(m.weight)
            elif weight_init == 'orthogonal': nn.init.orthogonal_(m.weight)
            elif weight_init == 'xavier_uniform': nn.init.xavier_uniform_(m.weight)
            elif weight_init == 'xavier_normal': nn.init.xavier_normal_(m.weight)
            elif weight_init == 'kaiming_uniform': nn.init.kaiming_uniform_(m.weight)
            elif weight_init == 'kaiming_normal': nn.init.kaiming_normal_(m.weight)
            if torch.is_tensor(m.bias): m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Linear):
            if weight_init == 'normal': nn.init.normal_(m.weight)
            elif weight_init == 'orthogonal': nn.init.orthogonal_(m.weight)
            elif weight_init == 'xavier_uniform': nn.init.xavier_uniform_(m.weight)
            elif weight_init == 'xavier_normal': nn.init.xavier_normal_(m.weight)
            elif weight_init == 'kaiming_uniform': nn.init.kaiming_uniform_(m.weight)
            elif weight_init == 'kaiming_normal': nn.init.kaiming_normal_(m.weight)
            if torch.is_tensor(m.bias): nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    if weight_init == 'normal': torch.nn.init.normal_(param.data)
                    elif weight_init == 'orthogonal': torch.nn.init.orthogonal_(param.data)
                    elif weight_init == 'xavier_uniform': torch.nn.init.xavier_uniform_(param.data)
                    elif weight_init == 'xavier_normal': torch.nn.init.xavier_normal_(param.data)
                    elif weight_init == 'kaiming_uniform': torch.nn.init.kaiming_uniform_(param.data)
                    elif weight_init == 'kaiming_normal': torch.nn.init.kaiming_normal_(param.data)
    return network

def fix_random_seed(seed):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator

def save_checkpoint(state, is_best, file_folder, file_name='checkpoint.pth.tar'):
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))

def trivial_batch_collator(batch):
    return batch

def worker_init_reset_seed(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
