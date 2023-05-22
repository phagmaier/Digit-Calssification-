import numpy as np
import pandas as pd
import math
from scipy.special import expit
from scipy.special import expit

train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

train_labels = train['label'].values
test_labels = test['label'].values

train.drop('label', axis=1,inplace=True)
test.drop('label', axis=1,inplace=True)

X = train.values #train features
X_test = train.values #test features

y = train_labels #TRAIN labels
y_test = test_labels


class NN:
    def __init__(self,X=X,y=y,X_test=X_test,y_test=y_test,
                 data_size=100,output_size=10,
                lr = 0.01,iterations=100, momentum = .1,batches=None):
        self.batches = batches
        np.random.seed(42)
        self.momentum = momentum
        self.b1m = 1
        self.b2m = 1
        np.random.seed(42)
        self.testX = X_test[:data_size]
        self.testy = y_test[:data_size]
        self.X = X[:data_size]
        self.y = y[:data_size]
        self.size = len(self.X[0])
        self.output_size = output_size
        self.w1 = np.random.normal(0.0, pow(self.size, -0.5), (self.size, self.size))
        self.w2 = np.random.normal(0.0, pow(self.output_size, -0.5), (self.output_size, self.size))
        self.m1 = np.full(self.w1.shape, 1)
        self.m2 = np.full(self.w2.shape, 1)
        self.b1 = np.random.rand()
        self.b2 = np.random.rand()
        self.lr = lr
        self.iterations = iterations
        #LAMBDA FUNCTIONS
        self.sig = expit
        self.cost_func = np.vectorize(lambda x,y: .5 * (y-x)**2)
        self.deriv_cost_func = np.vectorize(lambda x,y: -1 *(y-x))
        self.sig_deriv_func = np.vectorize(lambda x: x * (1-x))
        
    def train(self):
        if self.batches and self.batches > 1:
            X,Y = self.batchify(self.X,self.y,self.batches)
            self.train_batches(X,Y)
        else:
            self.train_stochastic()
            
    def train_batches(self,X,Y):
        for i in range(self.iterations):
            answers = []
            total_cost = 0
            for batch,y_batch in zip(X,Y):
                w1_batch_update = 0
                w2_batch_update = 0
                b1_batch_update=0
                b2_batch_update = 0
                for x,y in zip(batch,y_batch):
                    x = x.reshape(1,-1)
                    layer_1,layer_2 = self.forward(x)
                    ys,cost = self.calc_cost(layer_2,y)
                    total_cost += cost
                    update_w2,update_w1,b2_update,b1_update = self.back(ys,x,layer_1,layer_2)
                    w2_batch_update += update_w2
                    w1_batch_update +=update_w1
                    b1_batch_update += b1_update
                    b2_batch_update += b2_update
                    answers.append(1) if layer_2.argmax() == y else answers.append(0)
                w2_batch_update/= self.batches
                w1_batch_update/= self.batches
                b1_batch_update/=self.batches
                b2_batch_update /= self.batches
                self.update(w2_batch_update,w1_batch_update,b2_batch_update,b1_batch_update)
                
                avg_cost = total_cost / self.size
                percent = (sum(answers) / len(answers)) * 100
            if (i+1) % 10 == 0:
                print(f"THE AVERAGE COST OF ITERATION {i+1}: {avg_cost}")
                print(f"PERCENTAGE OF CORRECT PREDICTIONS: {percent}%")
                print('-----------------------------------------------------')
    def train_stochastic(self):
        for i in range(self.iterations):
            answers = []
            total_cost = 0
            for x,y in zip(self.X,self.y):
                x = x.reshape(1,-1)
                layer_1,layer_2=self.forward(x)
                ys,cost = self.calc_cost(layer_2,y)
                total_cost += cost
                update_w2,update_w1,b2_update,b1_update = self.back(ys,x,layer_1,layer_2)
                self.update(update_w2,update_w1,b2_update,b1_update)
                answers.append(1) if layer_2.argmax() == y else answers.append(0)
            percent = (sum(answers) / len(answers)) * 100
            avg_cost = total_cost/len(self.data_size)
            if (i+1) % 10 == 0:
                print(f'THE AVERAGE COST FOR ITERATION {i+1}: {avg_cost}')
                print(f'THE PERCENTAGE OF CORRECT ANSWERS: {percent}%')
                print('-----------------------------------------------------')
            
    def forward(self,x):
        layer_1 = self.sig(x @ self.w1.T + self.b1)
        layer_2 = self.sig(layer_1 @ self.w2.T + self.b2)
        return layer_1,layer_2
        
    def calc_cost(self,layer_2,y):
        ys = np.array([1 if i == y else 0 for i in range(10)])
        costs = self.cost_func(layer_2,ys)
        cost = np.sum(costs)
        return ys,cost
    
    def back(self,ys,x,layer_1,layer_2):
        #WEIGHTS
        cost_deriv = self.deriv_cost_func(layer_2,ys)
        sig_deriv_1 = self.sig_deriv_func(layer_2)
        update_w2 = cost_deriv * sig_deriv_1 * layer_1.T
        A = cost_deriv * sig_deriv_1
        B = self.sig_deriv_func(layer_1)
        update_w1 = ((A @ self.w2) * B) * x.T
        b2_update = np.sum(cost_deriv * sig_deriv_1)
        b1_update = np.sum((A @ self.w2) * B)
        return update_w2,update_w1,b2_update,b1_update
    
    def update(self, update_w2, update_w1, b2_update, b1_update):
        self.b1m = self.momentum * self.b1m + (1 - self.momentum) * b1_update
        self.b2m = self.momentum * self.b2m + (1 - self.momentum) * b2_update
        self.m1 = self.momentum * self.m1 + (1 - self.momentum) * update_w1.T
        self.m2 = self.momentum * self.m2 + (1 - self.momentum) * update_w2.T
        self.w1 = self.w1 - self.lr * self.m1
        self.w2 = self.w2 - self.lr * self.m2
        self.b2 = self.b2 - self.lr * self.b2m
        self.b1 = self.b1 - self.lr * self.b1m
    
    def batchify(self,X, y, batch_size):
        assert len(X) == len(y), "Input arrays must have the same length."
        assert batch_size > 0, "Batch size must be a positive integer."

        num_samples = len(X)
        num_batches = int(np.ceil(num_samples / batch_size))

        batched_X = [np.array(x) for x in np.array_split(X, num_batches)]
        batched_y = [np.array(y) for y in np.array_split(y, num_batches)]

        return np.array(batched_X), np.array(batched_y)
        
        
    def testing(self):
        results = []
        for i,y in zip(self.testX,self.testy):
            layer_1,layer_2 = self.forward(i.reshape(1,-1))
            answer = layer_2.argmax()
            results.append(1) if answer == y else results.append(0)
        print('TESTING RESULTS')
        print('----------------------------------')
        print(f'Percentage of correct classification: {(sum(results)/len(results)) * 100}%')
        
        
