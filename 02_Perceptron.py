#import required libaries
import numpy as np
import time
from matplotlib import pyplot as plt

##### Perceptron Class #####
class Perceptron:
    def __init__(self, input_size, lr):
        # Hyperparameters
        self.lr = lr                        # learning rate attibute
        self.input_size    = input_size     # input layer attibute
        
        self.loss = []                      # cost list attribute
        self.y_hats = []                    # predictions list attribute
        
        # Save all weights
        self.initialize()
    
    def initialize(self):
        # init parameters
        self.weights = np.zeros(self.input_size)
        self.bias = 0
        
    # Function for forward propagation
    def forward(self, x):
        Z = np.dot(x, self.weights) + self.bias
        A = np.where(Z > 0, 1, 0)
        return Z, A

    # Back Propagation
    def backward(self, input, label, y_hat):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        
        dz = y_hat - label

        # compute gradients
        dW = dz * np.array(input)
        db = dz
        
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
    def fit(self, X, y, epochs=100):
        
        # Hyperparameters
        self.epochs = epochs

        # Train
        for epoch in range(self.epochs):
            # Iterate Pairs of Inputs and Labels
            for input, label in zip(X, y):
                # Forward
                Z, y_hat = self.forward(input)
                
                # Back Propagation
                dW, db = self.backward( input, label, y_hat)
                
                # Optimize / update weights and bias (gradient descent)
                self.update_params(dW, db)
                
        print('Training Complete')
        print('----------------------------------------------------------------------------')

    # Test
    def test(self, X, y):
        # Iterate Pairs of Inputs and Labels
        for input, label in zip(X, y):
            # Predict
            Z, y_hat = self.forward(input)

            # Print
            print(f'Input: {input}, Prediction: {y_hat}, Label: {label}')

# Initialize Perceptron
model = Perceptron(input_size = 2, lr = 0.01)

##### Training #####
training_inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
training_labels = [1, 0, 0, 0]

epochs       = 1000

model.fit(training_inputs, training_labels, epochs=1000)

##### Testing #####
testing_inputs = [[1, 1], [0, 1]]
testing_labels = [1, 0]
model.test(testing_inputs, testing_labels)
