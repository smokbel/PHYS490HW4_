import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    '''
    Variational Auto-encoder
    Encoding:
        1x1 Convolution layer to reduce complexity
        1 Maxpool
        Two fully-connected layers fc1 and fc2.
        Two ReLU activation functions
    
    Decoding:
        Two fully connected layers
        Activation functions - relu and sigmoid 
    '''

    def __init__(self):
        super(VAE, self).__init__()

        #Encoding layers
        self.encode = nn.Sequential()
        self.encode.add_module("Conv1", nn.Conv2d(1,1,3))
        self.encode.add_module("maxpool", nn.MaxPool2d(1))
        self.encode.add_module("relu1", nn.ReLU())
        self.encode.add_module("batch", nn.BatchNorm2d(1))

        #Decoding layers
        self.decode = nn.Sequential()
        self.decode.add_module("fc3", nn.Linear(20,80))
        self.encode.add_module("relu1", nn.ReLU())
        self.decode.add_module("fc4", nn.Linear(80,14*14))
        self.decode.add_module("sigmoid", nn.Sigmoid())

        #FC Linear functions
        self.fc1= nn.Linear(12*12, 80)
        self.fc21= nn.Linear(80, 20)
        self.fc22= nn.Linear(80, 20)

    #Encoder
    def encoder(self, x):
        h1 = self.encode(x)
        #correct dimensions for tensor to work
        h1= h1.view(h1.size(0), -1) 
        h1 = F.relu(self.fc1(h1))
        h12 = self.fc21(h1)
        h21 = self.fc22(h1)
        #Encoder outputs two vectors of means and stds
        return h12, h21

    #Common reparam function
    def reparamaterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, z):
        h3 = self.decode(z)
        #get into format of images
        h4 = h3.reshape([h3.size()[0],1,14,14])
        return h4

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparamaterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    #Loss function
    def loss_func(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, (x*1/256), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD