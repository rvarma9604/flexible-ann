#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MLP(nn.Module):
	def __init__(self, input_dim, output_dim, params):
		super(MLP, self).__init__()

		# decode params
		hidden_dims, p_drop, activation, batchnorm = self.decode(params)

		# layer-dimensions
		dlayers = [input_dim] + hidden_dims + [output_dim]

		# network definintion
		self.network = nn.Sequential()
		l=0
		for i in range(len(dlayers) - 2):
			self.network.add_module(
					name=str(l),
					module=nn.Linear(dlayers[i], dlayers[i + 1]))
			self.network.add_module(
					name=str(l + 1),
					module=nn.ReLU(inplace=True) if activation=='relu' else 
					(nn.Tanh() if activation=='tanh' else nn.Sigmoid()))
			self.network.add_module(
					name=str(l + 2),
					module=nn.Dropout(p_drop))
			if batchnorm:
				self.network.add_module(
					name=str(l + 3),
					module=nn.BatchNorm1d(dlayers[i + 1]))
				l += 1
			l += 3

		# last layer
		self.network.add_module(
				name=str(l),
				module=nn.Linear(dlayers[-2], dlayers[-1]))

	def forward(self, x):
		return self.network(x)

	def decode(self, params):
		# default values
		hidden_dims = []
		p_drop = 0
		activation = 'relu'
		batchnorm = False

		if 'hidden_dims' in params.keys():
			hidden_dims = params['hidden_dims']
		if 'p_drop' in params.keys():
			p_drop = params['p_drop']
		if 'activation' in params.keys():
			activation = params['activation']
		if 'batchnorm' in params.keys():
			batchnorm = params['batchnorm']

		return hidden_dims, p_drop, activation, batchnorm


class Data(Dataset):
	def __init__(self, dataset, features):
		self.len = len(dataset)
		self.x = torch.from_numpy(dataset.iloc[:, :features].values).float()
		self.y = torch.from_numpy(dataset.iloc[:, features].values).long()

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.len
