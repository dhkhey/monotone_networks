import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class monotone_network(nn.Module):
    def __init__(self, X,Y):
        super(monotone_network, self).__init__()
        self.X = X 
        self.Y = Y
        self.n = len(X) # number of data points
        self.d = len(X[0]) #number of dimensions

        self.l1 = nn.Linear(in_features = self.n, out_features = self.n*self.d)
        self.l1_initialize(self.n, self.d)

        self.l2 = nn.Linear(in_features =  self.n*self.d, out_features = self.n)
        self.l2_initialize(self.n,self.d)

        self.l3 = nn.Linear(in_features=self.n, out_features=self.n)
        self.l3_initialize(self.n)
        
    def l1_initialize(self,n,d):
        w = np.eye(d,d,0)
        w = np.tile(w,(n,1))
        w = torch.nn.Parameter(torch.Tensor(w))
        self.l1.weight = w
        self.l1.bias = torch.nn.Parameter(np.negative(self.X.flatten()))
        
    def l2_initialize(self,n,d):
        w = np.zeros(n*d)
        w = np.tile(w, (n,1))
        for i in range(len(w)):
            for j in range(d):
              w[i][d*i+j] = 1
        w = torch.nn.Parameter(torch.Tensor(w))
        self.l2.weight = w
        self.l2.bias = torch.nn.Parameter(torch.Tensor(np.full((n),-d)))

    def l3_initialize(self,n):
        w = np.ones((n,n))
        w = np.triu(w)
        w = torch.nn.Parameter(torch.Tensor(w))
        self.l3.weight = w
        self.l3.bias = torch.nn.Parameter(torch.Tensor(np.full((n),-1)))

    # non-neg -> 1, neg -> 0
    def threshold_activation(self,x):      
        x = (x>=0.0).float()
        return x

    def output_layer(self,x):
        a = np.zeros(len(self.Y))
        a[0] = self.Y[0]
        for i in range(1, len(self.Y)):
          a[i] = self.Y[i] - self.Y[i-1]
        return x@a

    def forward(self, x):
        x = self.l1(x)
        x = self.threshold_activation(x)
        x = self.l2(x)
        x = self.threshold_activation(x)
        x = self.l3(x)
        x = self.threshold_activation(x)
        
        return self.output_layer(x)

        