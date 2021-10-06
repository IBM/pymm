"""
Description: 
Author      : Andrew Wood
Author_email: dewood@bu.edu
License     : Apache License, Version 2.0
"""

# SYSTEM IMPORTS
from typing import List
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt
import torch.nn.functional as F


class Model_2Conv2FC(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = pt.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = pt.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = pt.nn.Dropout2d()
        self.fc1 = pt.nn.Linear(320, 50)
        self.fc2 = pt.nn.Linear(50, 10)

    def forward(self,
                x: pt.Tensor):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)

    def get_params(self) -> np.ndarray:
        params_list: List[np.ndarray] = list()
        for P in self.parameters():
            params_list.append(P.cpu().detach().numpy().reshape(-1))
        return np.hstack(params_list).reshape(-1,1)

    def set_params(self,
                   theta: np.ndarray) -> None:
        param_idx: int = 0
        for P in self.parameters():
            if len(P.size() > 0):
                num_params: int = np.prod(P.size())
                P.copy_(theta[param_idx:param_idx+num_params]
                    .reshape(P.size()))
                param_idx += num_params

class Model_2FC(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = pt.nn.Linear(28*28, 1000)
        self.fc2 = pt.nn.Linear(1000, 10)

    def forward(self,
                x: pt.Tensor):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, -1)

    def get_params(self) -> np.ndarray:
        params_list: List[np.ndarray] = list()
        for P in self.parameters():
            params_list.append(P.cpu().detach().numpy().reshape(-1))
        return np.hstack(params_list).reshape(-1,1)

    def set_params(self,
                   theta: np.ndarray) -> None:
        param_idx: int = 0
        for P in self.parameters():
            if len(P.size() > 0):
                num_params: int = np.prod(P.size())
                P.copy_(theta[param_idx:param_idx+num_params]
                    .reshape(P.size()))
                param_idx += num_params


class Model_3FC(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = pt.nn.Linear(28*28, 1000)
        self.fc2 = pt.nn.Linear(1000, 500)
        self.fc3 = pt.nn.Linear(500, 10)

    def forward(self,
                x: pt.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, -1)

    def get_params(self) -> np.ndarray:
        params_list: List[np.ndarray] = list()
        for P in self.parameters():
            params_list.append(P.cpu().detach().numpy().reshape(-1))
        return np.hstack(params_list).reshape(-1,1)

    def set_params(self,
                   theta: np.ndarray) -> None:
        param_idx: int = 0
        for P in self.parameters():
            if len(P.size() > 0):
                num_params: int = np.prod(P.size())
                P.copy_(theta[param_idx:param_idx+num_params]
                    .reshape(P.size()))
                param_idx += num_params

class Model_4FC(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = pt.nn.Linear(28*28, 1000)
        self.fc2 = pt.nn.Linear(1000, 1000)
        self.fc3 = pt.nn.Linear(1000, 1000)
        self.fc4 = pt.nn.Linear(1000, 10)

    def forward(self,
                x: pt.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, -1)

    def get_params(self) -> np.ndarray:
        params_list: List[np.ndarray] = list()
        for P in self.parameters():
            params_list.append(P.cpu().detach().numpy().reshape(-1))
        return np.hstack(params_list).reshape(-1,1)

    def set_params(self,
                   theta: np.ndarray) -> None:
        param_idx: int = 0
        for P in self.parameters():
            if len(P.size() > 0):
                num_params: int = np.prod(P.size())
                P.copy_(theta[param_idx:param_idx+num_params]
                    .reshape(P.size()))
                param_idx += num_params

