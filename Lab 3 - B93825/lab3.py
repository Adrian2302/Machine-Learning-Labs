# Lab 3 - B93825 Adrián Hernández Young

import pandas as pd
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import train_test_split


def MSE(y_true, y_predict):
    return np.sum((y_true-y_predict)**2)/y_true.shape[0]

def score(y_true, y_predict):
    mean = y_true.mean()
    SST = ((y_true - mean)**2).sum()
    SSE = ((y_true - y_predict)**2).sum()
    return (SST - SSE) / SST


def MAE(y_true, y_predict):
    return np.sum((y_true-y_predict))/y_true.shape[0]


class LinearRegression:

    def __init__(self):
        self.weights = None

    def fit(self, x, y, max_epochs=100, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='none', lambd=0):
        n_samples, n_features = x.shape

        rng = default_rng()
        self.weights = rng.choice(1000, size=n_features+1, replace=False)

        bias = np.ones((n_samples,1))
        x = np.hstack((bias,x))

        epoch = 0
        old_de = 0
        actual_de = 0

        new_error = 5.0

        while epoch < max_epochs or new_error > threshold:
            xc = np.matmul(x,self.weights)
            
            if error == "mse":
                if regularization == "l1" or regularization == "lasso":
                    mse = MSE(y, xc)
                    #print(mse)
                    lasso_sum = lambd * np.sum(abs(self.weights))
                    actual_de = -2/n_samples * (np.matmul(y - xc, x)) + lasso_sum             

                elif regularization == "l2" or regularization == "ridge":
                    mse = MSE(y, xc)
                    #print(mse)
                    ridge_sum = lambd * np.sum(self.weights**2)
                    actual_de = -2/n_samples * (np.matmul(y - xc, x)) + ridge_sum            

                else: # none
                    mse = MSE(y, xc)
                    #print(mse)
                    actual_de = -2/n_samples * (np.matmul(y - xc, x))

            ''' INTENTO MAE
            else: # error == mae
                if regularization == "l1" or regularization == "lasso":   
                    mae = MAE(y, xc)
                    #print(mae)
                    lasso_sum = lambd * np.sum(abs(self.weights))
                    actual_de = -1/(n_samples * x) + lasso_sum
                    

                elif regularization == "l2" or regularization == "ridge":  
                    mae = MAE(y, xc)
                    #print(mae)
                    ridge_sum = lambd * np.sum(self.weights**2)
                    actual_de = -1/(n_samples * x) + ridge_sum
                
                else: # none 
                    mae = MAE(y, xc)
                    #print(mae)
                    actual_de = (-1/n_samples) * x
            '''

            self.weights = self.weights - learning_rate * (actual_de + momentum * old_de)

            if epoch > 1:
                new_error = np.sum(actual_de) - np.sum(old_de)
            old_de = actual_de

            learning_rate = learning_rate / (1 + decay)

            epoch += 1
            
        #print("MSE:", mse)

    def predict(self, x):
        bias = np.ones((x.shape[0],1))
        x = np.hstack((bias,x))
        return pd.Series(np.matmul(x,self.weights))

if __name__ == "__main__":
    
    df = pd.read_csv('fish_perch.csv')
    X = df.drop('Weight', axis=1)
    y = pd.Series(df['Weight'].to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=21)
    

# 5.a y 5.b

    print("\nNone")

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=100, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='none', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 100 - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=1000, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='none', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 1000 - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='none', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000 - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0.01, decay=0, error='mse', regularization='none', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Momentum - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0.01, error='mse', regularization='none', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Decay - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0.01, decay=0.01, error='mse', regularization='none', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Momentum, +Decay - Score:", score(y_test, predictions))



    print("\nLasso")

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=100, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='l1', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 100 - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=1000, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='l1', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 1000 - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='l1', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000 - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0.01, decay=0, error='mse', regularization='l1', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Momentum - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0.01, error='mse', regularization='l1', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Decay - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0.01, decay=0.01, error='mse', regularization='l1', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Momentum, +Decay - Score:", score(y_test, predictions))



    print("\nRidge")

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=100, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='l2', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 100 - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=1000, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='l2', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 1000 - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0, error='mse', regularization='l2', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000 - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0.01, decay=0, error='mse', regularization='l2', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Momentum - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0, decay=0.01, error='mse', regularization='l2', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Decay - Score:", score(y_test, predictions))

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0.01, decay=0.01, error='mse', regularization='l2', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Momentum, +Decay - Score:", score(y_test, predictions))



    print("\nPruebas extra")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=10)
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0.01, decay=0.01, error='mse', regularization='l2', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Momentum, +Decay - Score:", score(y_test, predictions))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=30)
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0.01, decay=0.01, error='mse', regularization='l2', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Momentum, +Decay - Score:", score(y_test, predictions))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train, max_epochs=10000, threshold=0.01, learning_rate=0.0001, momentum=0.01, decay=0.01, error='mse', regularization='l2', lambd=0)
    predictions = linear_regression.predict(X_test)
    print("Error = MSE, iter = 10000, +Momentum, +Decay - Score:", score(y_test, predictions))

    


'''

5)

c. ¿Cuál fue la combinación de parámetros que le proveyó el mejor resultado?

    R/ La combinación de parámetros que me produjo mejores resultados fue utilizando MSE, Ridge,
       100000 de max_epochs, con Momentum y Decay . Con un score de 0.8117834557764284.

d. ¿Qué pasa si utiliza esa misma combinación pero cambia la semilla del train_test_split? 
    Pruebe con varias semillas

    R/ Me ocurrió que en 2 corridas pegó valores muy bajos, mientras que en la otra muy alto.

e. Si pasa algo inusual: ¿Por qué cree que pasa esto?

    R/ Puede ser que en lo anterior que algunos casos sea preferible no utilizar ciertas variables, mientras
       que en otros casos sí.  También más en general, noté que si mi learning rate era 0.001, entonces
       tenía problemas para converger, tiraba problemas e infinito incluso cuando el learning rate era mayor.

'''