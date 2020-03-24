import sys
import argparse
import json
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from auto_encode import VAE
from data_gen import Data

#Command line arguments
parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('-param', type=str, default='param/param.json',
                    help='Enter filepath of json parameters. Current: param/param.json')
parser.add_argument('-o', type=str, default='results',
                    help='Result directory')
parser.add_argument('-n', type=int, default=100,
                    help='Enter number of images to create. Current: 100')

args = parser.parse_args()

if __name__ == '__main__':

    #Load param file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    # Load model and data parameters
    model= VAE()
    data= Data(param['data_file'],0)
    learning_rate = param['learning_rate']
    num_epochs= int(param['num_epochs'])
    display_epoch = param['display_epochs']
    batch = param['batch']

    data_size = len(data.x_train)
    n_samples = args.n

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses= []

    # Training. Reference: https://github.com/pytorch/examples/blob/master/vae/main.py
    for epoch in range(1, num_epochs + 1):
        
        model.train()
        train_loss = 0
        i = 0
        while i<data_size:
            x = data.x_train[i:(i+batch),:,:,:]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = model.loss_func(recon_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            i += batch

        losses.append(train_loss)
        
        if ((epoch + 1)%display_epoch)==0:
            print('Epoch: [{}/{}]'.format(epoch+1, num_epochs)+\
                  '\tTraining Loss: {:.4f}'.format(train_loss))

    #Final training loss
    print('Final training loss: {:.4f}'.format(losses[-1]))

    #Plots saved in results folder
    plt.plot(range(num_epochs), losses, color="red")
    plt.title('Training loss vs. epoch')
    plt.savefig('results/loss.pdf')
    plt.close()

    with torch.no_grad():
        sample = torch.randn(n_samples, 20)
        sample = model.decoder(sample)
        for i in range(n_samples):
            sample_n = sample[i,:,:,:]    
            save_image(sample_n.view(14, 14),
                    'results/%i.pdf'%(i+1))
