# Adrián Hernández Young - B93825
'''

Programa para cargar el .pt generado y calcular el accuracy del modelo. Luego crea y muestra
la matrtiz de confusión del modelo.

'''

import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

DIRECTORY = "images"

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

model = ConvNN()
checkpoint = torch.load('tensor.pt')
model.load_state_dict(checkpoint)

classes, x_train, y_train, x_test,y_test = load_data()

x_test = np.resize(x_test, (10000, 3, 32, 32))
x_test = torch.tensor(data=x_test, dtype=torch.float32)
y_test = torch.tensor(data=y_test, dtype=torch.float32)
y_pred = model(x_test)
y_pred = y_pred.detach().numpy()

num_correctos = 0
for i in range(len(y_test)):
    prediction = np.argmax(y_pred[i])
    print(f'Prediction: {prediction} Actual: {y_test[i]}')
    if prediction == y_test[i]:
        num_correctos += 1
accuracy = num_correctos/len(y_test)
print(f'Accuracy: {accuracy}')

prediction = [np.argmax(p) for p in y_pred]
labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CM = confusion_matrix(y_test, prediction)
sns.heatmap(CM, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.show()


'''

7. Accuracy: 0.6115 = 61%

8. 
¿Cuáles categorías confunde su red? 

    R/ Según mi red neuronal, los perros no existen, nunca clasificó algo como un perro. 
       Confundió demasiado a perros como si fueran gatos, muchos pájaros como si fueran 
       gatos, venados como si fueran gatos, confundió ranas como si fueran gatos y carros 
       como si fueran camiones.

¿Por qué cree que esas categorías le generan confusión/errores de clasificación?
    
    R/ Con la parte de confundir carros como si fueran camiones es entendible, ambos
       son parecidos, 4 ruedas, puertas, etc, por lo que pueden generar que al momento
       de analizar las imágenes tengan características similares y clasifique dichas
       imágenes con valores parecidos. 
       Con respecto a los gatos, no estoy muy seguro de por qué confunde tantos valores
       como si fueran gatos. Tal vez con los venados, caballos y perros puede ser que 
       como todos tienen 4 patas, colas, y pelajes parecidos se pueda confundir.
       Pero por el lado de los pájaros y ranas con gatos no estoy muy seguro, ¿los ojos?.

'''
