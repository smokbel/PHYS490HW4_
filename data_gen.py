import numpy as np
import torch

class Data():
    def __init__(self, data_path, n_test_data):
        '''
        Reshape arrays into X: 14x14 MNIST image and its label,
        Y: one-hot vector representing the number by the index/2.
        
        ''' 
        data = np.loadtxt(data_path)
        n_total_data = len(data)
 
        ## The labels are the last element of each array. So, the y values will be the last element of each array
        ## and the x values are the 14x14 matrix corresponding to the label.
        
        def y_to_vector(length):
            return np.zeros([length,1,5])
        
        train_len = n_total_data - n_test_data
        
        y_test = y_to_vector(n_test_data)
        y_train = y_to_vector(train_len)
        
        y_test_value = data[:n_test_data,-1]/2
        x_test = data[:n_test_data,:-1].reshape(-1,1,14,14)
        y_train_value = data[n_test_data:,-1]/2
        x_train = data[n_test_data:,:-1].reshape(-1,1,14,14)
        
        ##the y values will correspond to a 5x1 vector so that there is 
        ## an exact solution for each possible label. This is known as 
        ## the one-hot vector

        for i in range(n_test_data):
            y_test[i-1,0,int(y_test_value[i-1])] = 1
            
        for i in range(train_len):
            y_train[i-1,0,int(y_train_value[i-1])] = 1
        
        # in order for the encoder to work, must make into torch tensor
        x_train= torch.tensor(np.array(x_train, dtype= np.float32))
        y_train= torch.tensor(np.array(y_train, dtype= np.float32))
        y_test = torch.tensor(np.array(y_test, dtype= np.float32))
        x_test = torch.tensor(np.array(x_train, dtype= np.float32))

        self.y_test = y_test
        self.x_test = x_test
        self.y_train = y_train
        self.x_train = x_train 
       