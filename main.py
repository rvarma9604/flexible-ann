import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from helper import *
from models import MLP, Data

'''dataset related parameters'''
file_name = "iris_dataset.txt"
features = 4
num_class = 3

'''model related parameters'''
params = {
	'hidden_dims': [6, 5],
	'p_drop': 0,
	'activation': 'relu',
	'batchnorm': True
}
epochs = 50

'''load and split the dataset'''
dataset = load_csv(file_name, features, adjust=True, header=None)
train, test = train_test_split(dataset, test_size=0.1, stratify = dataset.iloc[:, features], random_state=1)
trainer = Data(train, features)
tester = Data(test, features)

'''model setup'''
model = MLP(features, num_class, params)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print(f'The Model:\n{model}')

'''train the model'''
train_loss, train_acc, test_loss, test_acc =\
		model_train(model,
					trainer,
					optimizer,
					criterion,
					tester=tester,
					batch_size=10,
					epochs=epochs)

'''plot the performance'''
performance_plot(train_loss, test_loss, 0.5, "loss", "Loss.jpeg")
performance_plot(train_acc, test_acc, 0.5, "accuracy", "Accuracy.jpeg")

'''test output'''
xs, ys = tester[:]
model.eval()
yhats = model(xs).softmax(axis=1).argmax(axis=1)
yhats = yhats.detach().numpy()
df = pd.DataFrame({"True labels": ys, "Predicted labels": yhats}).to_string(index=False)
print(f'\nTest Output: \n{df}')