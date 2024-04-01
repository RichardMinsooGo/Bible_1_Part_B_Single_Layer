#import required libaries
import numpy as np
import time
from matplotlib import pyplot as plt

class LinearRegression:

    print('Running...')
    
    def __init__(self, input_size, lr: int = 0.01) -> None:
        # Hyperparameters
        self.lr = lr                        # learning rate attibute
        self.input_size    = input_size     # input layer attibute
        
        
        # Save all weights
        self.initialize()
    
    def initialize(self):
        # init parameters
        self.weights = np.random.rand(self.input_size)  # W shape [f, 1]
        self.bias = 0
        
    # Function for forward propagation
    def forward(self, x):
        Z = np.dot(x, self.weights) + self.bias
        return Z

    # Back Propagation
    def backward(self, X_train, y_train, y_hat):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        
        # number of examples
        n_samples, n_features = X_train.shape          # X shape [N, f]
        
        dz = y_hat - y_train # derivative of sigmoid and bce X_train.T*(A-y)

        # compute gradients
        dW = (1 / n_samples) * np.dot(X_train.T, dz)
        db = (1 / n_samples) * np.sum(dz)
        
        return dW, db
    
    def MSELoss(self, y_true, y_pred): # Mean Square cost function
        # use mean() and square() methods
        MSE = np.mean(np.square(y_true - y_pred))
        return MSE

    def update_params(self, dW, db):
        '''
            Stochatic Gradient Descent (SGD):
            θ^(t+1) <- θ^t - η∇L(y, ŷ)

            Momentum:
            v^(t+1) <- βv^t + (1-β)∇L(y, ŷ)^t
            θ^(t+1) <- θ^t - ηv^(t+1)
        '''
        self.weights -= self.lr * dW
        self.bias    -= self.lr * db
        
    # Train
    def fit(self, X, y, epochs=1000):
        
        # Hyperparameters
        self.epochs = epochs
        # num_samples, num_features = X.shape          # X shape [N, f]

        # Train
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}')
            
            # Forward
            # y_hat shape should be N, 1
            y_hat = self.forward(X)
            
            # Back Propagation
            dW, db = self.backward(X, y, y_hat)
            
            # Optimize / update weights and bias (gradient descent)
            self.update_params(dW, db)
                
        print('Training Complete')
        print('----------------------------------------------------------------------------')

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

        
from sklearn import datasets, metrics, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = datasets.make_regression(
    n_samples=500, n_features=1, noise=15, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

def plot_data(x, y):
    plt.xlabel('house size')
    plt.ylabel('price')
    plt.plot(x[:,0], y, 'bo')
    plt.show()

plot_data(X, y)

input_size = 1

model = LinearRegression(input_size)
plt.xlabel('house size')
plt.ylabel('price')
plt.plot(X[:,0], y, 'bo')
plt.plot(X, LinearRegression(input_size).fit(X, y, epochs=0).predict(
    X), linewidth=2, color='black', label='prediction')
plt.legend()
plt.show()


model.fit(X_train, y_train, epochs=1000)

predictions = model.predict(X_test)
print(f'MSE: {mean_squared_error(predictions, y_test)}')

# print(X.shape, y.shape)

plt.xlabel('house size')
plt.ylabel('price')
plt.plot(X[:,0], y, 'bo')

plt.plot(X, model.predict(X), linewidth=2,
          color='black', label='prediction')
plt.legend()
# plt.grid()
plt.show()

