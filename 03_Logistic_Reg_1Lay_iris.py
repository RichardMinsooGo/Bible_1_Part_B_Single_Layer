#import required libaries
import numpy as np
import time
from matplotlib import pyplot as plt

class LogisticRegression():

    """
    This is the LogisticRegression class used to feedforward and backpropagate the network across a defined number
    of iterations and produce predictions. After iteration the predictions are assessed using
    Binary Cross Entropy Cost function.
    """

    print('Running...')
    
    def __init__(self, lr=1e-1, input_size = 30):
        # Hyperparameters
        self.lr = lr                        # learning rate attibute
        self.input_size    = input_size     # input layer attibute
        
        self.loss = []                      # cost list attribute
        self.y_hats = []                    # predictions list attribute
        
        # Save all weights
        self.initialize()

    def sigmoid(self, x, derivative=False):
        '''
            Forward path:
            σ(x) = 1 / 1+exp(-z)

            Backward path:
            ∇σ(x) = exp(-z) / (1+exp(-z))^2
        '''
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))
    
    def initialize(self):
        # init parameters
        self.weights = np.zeros(self.input_size)
        self.bias = 0
        
    # Function for forward propagation
    def forward(self, x):
        Z = np.dot(x, self.weights) + self.bias
        y_hat = self.sigmoid(Z, derivative=False)  # output layer prediction
        return Z, y_hat

    # Back Propagation
    def backward(self, Z, X_train, y_train, y_hat):
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
        n_samples, n_features = X_train.shape
        
        dz = y_hat - y_train # derivative of sigmoid and bce X_train.T*(A-y)

        # compute gradients
        dW = (1 / n_samples) * np.dot(X_train.T, dz)
        db = (1 / n_samples) * np.sum(dz)
        
        return dW, db
    
    def BCELoss(self, y_true, y_pred): # binary cross entropy cost function
        # binary cross entropy
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

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
        
    def check_accuracy(self, y_true, y_pred):
        pred_labels = y_pred > 0.5
        accuracy = np.sum(y_true == pred_labels) / len(y_true)
        return accuracy
    # Train
    def fit(self, X_train, y_train, epochs=100):
        
        # Hyperparameters
        self.epochs = epochs
        
        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}"

        # Train
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}')
            
            # Forward
            Z, y_hat = self.forward(X_train)

            # Evaluate performance
            # Training data
            train_acc  = self.check_accuracy(y_train, y_hat)
            train_loss = self.BCELoss(y_train, y_hat)
            
            # store BCE cost in list
            self.loss.append(train_loss)
            
            # Back Propagation
            dW, db = self.backward(Z, X_train, y_train, y_hat)
            
            # Optimize / update weights and bias (gradient descent)
            self.update_params(dW, db)
            
            print(template.format(epoch+1, time.time()-start_time, train_acc, train_loss))
                
        print('Training Complete')
        print('----------------------------------------------------------------------------')

from sklearn import datasets, metrics, model_selection, preprocessing
# Load data
iris = datasets.load_iris()

X = np.array(iris.data[:100])
y = np.array(iris.target[:100])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1234)


epochs       = 1000

input_size   = X_train.shape[1]

# Print data shape
print("Train/Test/Validation Data Shape")
print("X_train.shape :", X_train.shape)
print("X_test.shape  :", X_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)

model = LogisticRegression(lr=0.01, input_size = input_size)
model.fit(X_train, y_train, epochs=epochs)

# plot the cost function
plt.grid()
plt.plot(range(model.epochs),model.loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('BCE Loss Function')
plt.show()

