#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_csv(file_name, features, adjust=False, header=None):
	'''
	Function to load the csv file
	
	Inputs
	------
	file_name: path to the csv file
	features: number of feature variables in our dataset
	adjust: convert class labels from {1,...,n} to {0,...,n-1}
			(if single output variable exists)
	header: if the csv file has a header

	Returns
	-------
	datased: pandas dataframe
	'''

	dataset = pd.read_csv(file_name, header=header)
	if adjust:
		dataset.iloc[:, features] = dataset.iloc[:, features] - 1
	return dataset

def train(model, train_loader, criterion, optimizer, device):
	'''
	Function to perform one training epoch
	'''
	model.train()
	train_loss, train_acc = 0, 0
	for x, y in train_loader:
		x, y = x.to(device), y.to(device)
		yhat = model(x).softmax(axis=1)
		loss = criterion(yhat, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		yhat = yhat.argmax(axis=1)
		train_acc += (yhat == y).sum()
		train_loss += loss.item()
	train_acc = ((train_acc * 100.0) / len(train_loader.dataset)).item()
	train_loss = train_loss / len(train_loader.dataset)

	print(f'\tTrain Loss: {train_loss:.4f}\tTrain Accuracy: {train_acc:.4f}%')
	return train_loss, train_acc

def test(model, test_loader, criterion, device):
	'''
	Function to perform one test epoch
	'''
	model.eval()
	test_loss, test_acc = 0, 0
	with torch.no_grad():
		for x, y in test_loader:
			x, y = x.to(device), y.to(device)
			yhat = model(x).softmax(axis=1)
			loss = criterion(yhat, y)

			yhat = yhat.argmax(axis=1)
			test_acc += (yhat == y).sum()
			test_loss += loss.item()
	test_acc = ((test_acc * 100.0) / len(test_loader.dataset)).item()
	test_loss = test_loss / len(test_loader.dataset)

	print(f'\tTest Loss: {test_loss:.4f}\tTest Accuracy: {test_acc:.4f}%')
	return test_loss, test_acc			

def model_train(model, trainer, optimizer, criterion, tester=None, batch_size=10, epochs=10):
	'''
	Function to train the model

	Inputs
	------
	model
	trainer: Data variable loaded with training set
	optimizer
	criterion 
	tester: Data variable loaded with test(or validation) set
	batch_size: mini-batch_size
	epochs

	Returns
	-------
	train_plt: list of train loss at each epoch
	test_plt: list of test loss at each epoch (if tester present)
	'''
	train_loader = DataLoader(trainer, batch_size=batch_size, shuffle=True)
	if tester:
		test_loader = DataLoader(tester, batch_size=batch_size, shuffle=False)

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		print("\nUsing GPU")
	else:
		device = torch.device("cpu")
		print("\nUsing CPU")


	train_loss_plt, train_acc_plt = [], []
	test_loss_plt, test_acc_plt = [], []
	print("\nTraining")
	model.to(device)
	for epoch in range(epochs):
		print(f'Epoch {epoch + 1}/{epochs}:')
		train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
		train_loss_plt.append(train_loss), train_acc_plt.append(train_acc)
		if tester:
			test_loss, test_acc = test(model, test_loader, criterion, device)
			test_loss_plt.append(test_loss), test_acc_plt.append(test_acc)

	model.to("cpu")
	return (train_loss_plt, train_acc_plt, test_loss_plt, test_acc_plt) if tester else (train_loss_plt, train_acc_plt)

def smoothen(values, weight=0.7):
	'''
	Function to smoothen out the values of the list to plot the general trend
	
	Inputs
	------
	values: list of numbers
	weight: smoothening parameter - [0,1)

	Returns
	-------
	smoothed: smoothened out list of numbers
	'''
	last = values[0]
	smoothed = []
	for val in values:
		smoothed_val = last * weight + (1 - weight) * val
		smoothed.append(smoothed_val)
		last = smoothed_val
	return smoothed

def performance_plot(train_plt, test_plt=None, smoothening=0.7, ylabel="loss", file_name="performance.jpg"):
	'''
	Function to plot the model's performance

	Inputs
	------
	train_plt: list of train loss at different epochs
	test_plt: list of test loss at different epochs
	smoothening: weight associated with the smoothen function
	file_name: path to store the plot
	'''
	plt.figure(figsize=(10,8))
	plt.plot(smoothen(train_plt, smoothening), label="train")
	if test_plt:
		plt.plot(smoothen(test_plt, smoothening), label="test")
	plt.legend()
	plt.xlabel("epochs")
	plt.ylabel(ylabel)
	plt.title("Performance")
	plt.savefig(file_name, dpi=240) 