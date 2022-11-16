# Adrián Hernández Young - B93825
'''

Programa para generar el modelo de la red neuronal y guardarlo en el archivo .pt.

'''

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Custom subdirectory to find images
DIRECTORY = "images"

# Método usado para separar el data en batches más perqueños, sino me explota por memoria.
class Dset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

def load_data():
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    names = [n.decode('utf-8') for n in unpickle(DIRECTORY+"/batches.meta")[b'label_names']]
    x_train = None
    y_train = []
    for i in range(1,6):
        data = unpickle(DIRECTORY+"/data_batch_"+str(i))
        if i>1:
            x_train = np.append(x_train, data[b'data'], axis=0)
        else:
            x_train = data[b'data']
        y_train += data[b'labels']
    data = unpickle(DIRECTORY+"/test_batch")
    x_test = data[b'data']
    y_test = data[b'labels']
    return names,x_train,y_train,x_test,y_test

def plot_tensor(tensor, perm=None):
    if perm==None: perm = (1,2,0)
    plt.figure()
    plt.imshow(tensor.permute(perm).numpy().astype(np.uint8))
    plt.show()

# Clase de la red neuronal convolucional
class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), -1)
        return x

names,x_train,y_train,x_test,y_test = load_data()

x_train = np.resize(x_train, (50000, 3, 32, 32))
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

print(x_train.shape)
print(y_train.shape)
train = Dset(x_train, y_train)

x_test = np.resize(x_test, (10000, 3, 32, 32))
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

batch_split = DataLoader(dataset=train, batch_size=100)

model = ConvNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1, rho=0.9)

# Entrenamiento
for i in range(10):
    for i, (input, label) in enumerate(batch_split):
        optimizer.zero_grad()			# Coloca los Δw en 0
        y_pred = model(input)		    # Predice los valores del conjunto de entrenamiento
        label = label.type(torch.LongTensor)
        loss = loss_fn(y_pred, label)	# Calcula la pérdida
        loss.backward()				    # Calcula el backprogration (Δw) y acumula el error
        optimizer.step()				# Aplica los Δw acumulados y avanza un paso la iter. 
        print(f'Loss {i}: {loss.item()}') # Imprime la pérdida 

# Guardar el modelo
torch.save(model.state_dict(), 'tensor.pt')

